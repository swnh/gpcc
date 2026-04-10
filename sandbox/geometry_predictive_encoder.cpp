/* The copyright in this software is being made available under the BSD
 * Licence, included below.  This software may be subject to other third
 * party and contributor rights, including patent rights, and no such
 * rights are granted under this licence.
 *
 * Copyright (c) 2020, ISO/IEC
 * All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *
 * * Redistributions of source code must retain the above copyright
 *   notice, this list of conditions and the following disclaimer.
 *
 * * Redistributions in binary form must reproduce the above copyright
 *   notice, this list of conditions and the following disclaimer in the
 *   documentation and/or other materials provided with the distribution.
 *
 * * Neither the name of the ISO/IEC nor the names of its contributors
 *   may be used to endorse or promote products derived from this
 *   software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
 * ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
 * LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
 * CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
 * SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
 * INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
 * CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
 * ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
 * POSSIBILITY OF SUCH DAMAGE.
 */

#include "geometry_predictive.h"
#include "geometry.h"
#include "pointset_processing.h"
#include "quantization.h"

#include "PCCMisc.h"

#include "nanoflann.hpp"
#include <algorithm>
#include <bitset>
#include <iomanip>
#include <iostream>

namespace pcc {

//============================================================================

namespace {
  struct NanoflannCloud {
    std::vector<Vec3<int32_t>> pts;

    inline size_t kdtree_get_point_count() const { return pts.size(); }

    int32_t kdtree_get_pt(const size_t idx, const size_t dim) const
    {
      return pts[idx][dim];
    }

    template<class BBOX>
    bool kdtree_get_bbox(BBOX& /* bb */) const
    {
      return false;
    }
  };
}  // namespace

//============================================================================

namespace {
  float estimate(bool bit, AdaptiveBitModel& model)
  {
    return -log2(dirac::approxSymbolProbability(bit, model) / 128.);
  }
}  // namespace

//============================================================================

// Max ring groups for context array sizing (compile-time)
static const int kMaxRingGroups = 32;
static const int kNumRings = 32;
static const int kHistSize = 4;
static const int kHistTestCount = 3;
static const int kCartesianRangeClasses = 3;
static const int kCartesianBoundaryClasses = 2;
static const int kCartesianPredModes = 9;
static const int kCartesianPredFamilies = 3;
static const bool kEnableModePenaltyAnalysis = false;

namespace {

enum CartesianPredMode
{
  kPredZero = 0,
  kPredSameRingLast = 1,
  kPredSameRingLinear2 = 2,
  kPredCrossRingUp = 3,
  kPredCrossRingDown = 4,
  kPredFusedPrevPhi = 5,
  kPredSameRingCubic3 = 6,
  kPredCrossRingCorrected = 7,
  kPredSameRingHist = 8,
};

inline int
clampLaserIdx(const PCCPointSet3& cloud, int idx)
{
  if (!cloud.hasLaserAngles())
    return std::max(0, std::min(kNumRings - 1, idx));
  return std::max(0, std::min(kNumRings - 1, cloud.getLaserAngle(idx)));
}

inline int
computeRApprox(const point_t& pt)
{
  auto absX = std::abs(pt[0]);
  auto absY = std::abs(pt[1]);
  return std::max(absX, absY) + ((424 * std::min(absX, absY)) >> 10);
}

inline int
computeRangeClass(int rApprox)
{
  if (rApprox < 1 << 11)
    return 0;
  if (rApprox < 1 << 15)
    return 1;
  return 2;
}

inline int
modePenalty(int mode)
{
  static const int kModePenalty[kCartesianPredModes] = {
    1, 0, 0, 0, 0, 0, 0, 0, 0};
  return kModePenalty[mode];
}

inline float
quantile(std::vector<float> values, float q)
{
  if (values.empty())
    return 0.0F;
  q = std::max(0.0F, std::min(1.0F, q));
  std::sort(values.begin(), values.end());
  const size_t idx = size_t(q * float(values.size() - 1));
  return values[idx];
}

inline int
boundaryThreshold()
{
  return 2048;
}

inline int
predFamily(int mode)
{
  switch (mode) {
  case kPredSameRingLast:
  case kPredSameRingLinear2:
  case kPredSameRingCubic3:
  case kPredSameRingHist:
    return 0;
  case kPredCrossRingUp:
  case kPredCrossRingDown:
  case kPredCrossRingCorrected:
    return 1;
  default:
    return 2;
  }
}

}  // namespace

class PredGeomEncoder : protected PredGeomContexts {
public:
  PredGeomEncoder(const PredGeomEncoder&) = delete;
  PredGeomEncoder& operator=(const PredGeomEncoder&) = delete;

  PredGeomEncoder(
    const GeometryParameterSet&,
    const GeometryBrickHeader&,
    const PredGeomEncOpts&,
    const PredGeomContexts& ctxtMem,
    EntropyEncoder* aec,
    int numRingGroups = 1);

  // NB: The following constraint must be honoured:
  //      qp % (1 << geom_qp_multiplier_log2) == 0
  int qpSelector(const GNode& node) const { return _sliceQp; }

  void encode(
    Vec3<int32_t>* cloudA,
    Vec3<int32_t>* cloudB,
    const GNode* nodes,
    int numNodes,
    int* codedOrder,
    PredGeomPredictor& refFrameSph,
    PredGeomPredictor& refFrameSph2,
    bool reversed);

  int encodeTree(
    Vec3<int32_t>* cloudA,
    Vec3<int32_t>* cloudB,
    const GNode* nodes,
    int numNodes,
    int rootIdx,
    int* codedOrder,
    PredGeomPredictor& refFrameSph,
    PredGeomPredictor& refFrameSph2);

  void encodeNumDuplicatePoints(int numDupPoints);
  void encodeNumChildren(int numChildren);
  void encodePredMode(GPredicter::Mode mode);
  void encodePredIdx(int predIdx);

  //==================================================
  void encodePredGeom(
    const Vec3<int32_t>& residual,
    int group,
    int mode,
    int rangeClass,
    bool boundary);
  void encodeMode(int mode, int group, int rangeClass, bool boundary);
  void encodeHistIdx(int histIdx, int group, int rangeClass, bool boundary);
  float estimateModeBits(int mode, int group, int rangeClass, bool boundary);
  float estimateHistIdxBits(int histIdx, int group, int rangeClass, bool boundary);
  float estimatePredGeomBits(
    const Vec3<int32_t>& residual,
    int group,
    int mode,
    int rangeClass,
    bool boundary);
  //==================================================

  void encodeResidual(const Vec3<int32_t>& residual, int iMode, int multiplier, int rPred, int predIdx, const bool interFlag
    , int refNodeIdx
  );
  void encodeResPhi(
    int32_t resPhi, int predIdx, int boundPhi, const bool interFlag
    , int refNodeIdx
  );
  void
  encodeResR(int32_t resR, int multiplier, int predIdx, const bool interFlag
    , int refNodeIdx
  );

  void encodeResidual2(const Vec3<int32_t>& residual);
  void encodePhiMultiplier(const int32_t multiplier, const bool interFlag
    , int refNodeIdx
    , int predIdx
  );
  void encodeInterFlag(const bool interFlag, const uint8_t interFlagBuffer
  );
  void encodeRefNodeIdx(int refNodeIdx, bool globalMotionEnabled);
  void encodeRefDirFlag(bool refDirFlag);
  void encodeQpOffset(int dqp);
  void encodeEndOfTreesFlag(int endFlag);


  template<size_t NumPrefixCtx, size_t NumSuffixCtx>
  inline float
  estimateExpGolomb(
    unsigned int symbol,
    int k,
    AdaptiveBitModel (&ctxPrefix)[NumPrefixCtx],
    AdaptiveBitModel (&ctxSuffix)[NumSuffixCtx]);

  float estimateResPhi(int32_t resPhi, int predIdx, int boundPhi, const bool interFlag
    , int refNodeIdx
  );

  float estimateResR(
    int32_t resR, int multiplier, int predIdx, const bool interFlag
    , int refNodeIdx
  );

  float estimateBits(
    GPredicter::Mode mode,
    int predIdx,
    const Vec3<int32_t>& residual,
    int qphi,
    int rPred,
    bool interFlag,
    bool interEnabledFlag,
    int refNodeIdx,
    bool globalMotionEnabled,
    int numRef,
    bool refDirFlag,
    //bool refNodeFlag,
    const uint8_t interFlagBuffer,
    float best_known_bits);

  const PredGeomContexts& getCtx() const { return *this; }

  void setMinRadius(int value) { _pgeom_min_radius = value; }

private:
  EntropyEncoder* _aec;
  std::vector<int32_t> _stack;
  bool _geom_unique_points_flag;

  bool _geom_angular_mode_enabled_flag;
  bool _predgeometry_residual2_disabling_enabled_flag;
  Vec3<int32_t> origin;
  int _numLasers;
  SphericalToCartesian _sphToCartesian;
  bool _azimuth_scaling_enabled_flag;
  int _geomAngularAzimuthSpeed;

  bool _geom_scaling_enabled_flag;
  int _geom_qp_multiplier_log2;
  int _sliceQp;
  int _qpOffsetInterval;

  int _azimuthTwoPiLog2;

  Vec3<int> _maxAbsResidualMinus1Log2;
  Vec3<int> _pgeom_resid_abs_log2_bits;

  // Minimum radius used for prediction in angular coding
  int _pgeom_min_radius;

  int _maxPredIdx;
  int _maxPredIdxTested;
  int _thObj;
  int _thQphi;

  // ---- Per-ring-group context arrays (used by encodePredGeom) ----
  int _numRingGroups;  // runtime: 1, 2, 4, 8, 16, or 32
  AdaptiveBitModel _ctxResGt0_g[kMaxRingGroups][kCartesianRangeClasses]
                              [kCartesianBoundaryClasses][kCartesianPredModes][3];
  AdaptiveBitModel _ctxSign_g[kMaxRingGroups][kCartesianRangeClasses]
                             [kCartesianBoundaryClasses][kCartesianPredFamilies][3];
  AdaptiveBitModel _ctxNumBits_g[kMaxRingGroups][kCartesianRangeClasses]
                                [kCartesianBoundaryClasses][8][3][31];
  AdaptiveBitModel _ctxPredMode_g[kMaxRingGroups][kCartesianRangeClasses]
                                 [kCartesianBoundaryClasses][kCartesianPredModes];
  AdaptiveBitModel _ctxHistIdx_g[kMaxRingGroups][kCartesianRangeClasses]
                                [kCartesianBoundaryClasses][kHistSize - 1];
};

//============================================================================

PredGeomEncoder::PredGeomEncoder(
  const GeometryParameterSet& gps,
  const GeometryBrickHeader& gbh,
  const PredGeomEncOpts& opt,
  const PredGeomContexts& ctxtMem,
  EntropyEncoder* aec,
  int numRingGroups)
  : PredGeomContexts(ctxtMem)
  , _aec(aec)
  , _geom_unique_points_flag(gps.geom_unique_points_flag)
  , _geom_angular_mode_enabled_flag(gps.geom_angular_mode_enabled_flag)
  , _predgeometry_residual2_disabling_enabled_flag(gps.residual2_disabled_flag)
  , origin()
  , _numLasers(gps.numLasers())
  , _sphToCartesian(gps)
  , _azimuth_scaling_enabled_flag(gps.azimuth_scaling_enabled_flag)
  , _geomAngularAzimuthSpeed(gps.geom_angular_azimuth_speed_minus1 + 1)
  , _geom_scaling_enabled_flag(gps.geom_scaling_enabled_flag)
  , _geom_qp_multiplier_log2(gps.geom_qp_multiplier_log2)
  , _sliceQp(0)
  , _maxAbsResidualMinus1Log2((1 << gbh.pgeom_resid_abs_log2_bits) - 1)
  , _pgeom_resid_abs_log2_bits(gbh.pgeom_resid_abs_log2_bits)
  , _azimuthTwoPiLog2(gps.geom_angular_azimuth_scale_log2_minus11 + 12)
  , _pgeom_min_radius(gbh.pgeom_min_radius)
  , _maxPredIdx(gps.predgeom_max_pred_index)
  , _maxPredIdxTested(opt.maxPredIdxTested)
  , _thObj(gps.predgeom_radius_threshold_for_pred_list)
  , _thQphi(gps.resR_context_qphi_threshold)
  , _numRingGroups(numRingGroups)
{
  if (gps.geom_scaling_enabled_flag) {
    _sliceQp = gbh.sliceQp(gps);
    int qpIntervalLog2 =
      gps.geom_qp_offset_intvl_log2 + gbh.geom_qp_offset_intvl_log2_delta;
    _qpOffsetInterval = (1 << qpIntervalLog2) - 1;
  }

  if (gps.geom_angular_mode_enabled_flag)
    origin = gbh.geomAngularOrigin(gps);
  ;

    if (!gps.resR_context_qphi_threshold_present_flag){
    _thQphi = 0;
  }
  
  _stack.reserve(1024);
}

//----------------------------------------------------------------------------

void
PredGeomEncoder::encodeNumDuplicatePoints(int numDupPoints)
{
  _aec->encode(numDupPoints > 0, _ctxNumDupPointsGt0);
  if (numDupPoints)
    _aec->encodeExpGolomb(numDupPoints - 1, 0, _ctxNumDupPoints);
}

//----------------------------------------------------------------------------

void
PredGeomEncoder::encodeNumChildren(int numChildren)
{
  // Mapping order: 0, 1, 3, 2
  int val = numChildren ^ 1;

  _aec->encode(val > 0, _ctxNumChildren[0]);
  if (val > 0) {
    _aec->encode(val > 1, _ctxNumChildren[1]);
    if (val > 1)
      _aec->encode(val - 2, _ctxNumChildren[2]);
  }
}

//----------------------------------------------------------------------------

void
PredGeomEncoder::encodePredMode(GPredicter::Mode mode)
{
  int iMode = int(mode);
  _aec->encode((iMode >> 1) & 1, _ctxPredMode[0]);
  _aec->encode(iMode & 1, _ctxPredMode[1 + (iMode >> 1)]);
}

//-------------------------------------------------------------------------

void
PredGeomEncoder::encodePredIdx(int predIdx)
{
  for (int i = 0; i < predIdx; ++i)
    _aec->encode(1, _ctxPredIdx[i]);
  if (predIdx < _maxPredIdx)
    _aec->encode(0, _ctxPredIdx[predIdx]);
}
//----------------------------------------------------------------------------
void
PredGeomEncoder::encodeResR(int32_t resR, int multiplier, int predIdx, const bool interFlag
  , int refNodeIdx
)
{
  const int interCtx = interFlag ? 1 : 0;
  int ctxL = interFlag ? (refNodeIdx > 1 ? 1 : 0)
    : (predIdx ? 1 : 0);
  //int ctxL = predIdx == 0 /* parent */;
  int ctxLR = ctxL + (interFlag ? (abs(multiplier) > 2 ? 2 : 0)
    : (abs(multiplier) > _thQphi ? 2 : 0)); 
  //int ctxLR = ctxL + (multiplier ? 2 : 0);

  _aec->encode(resR != 0, _ctxResRGTZero[interCtx][ctxLR]);
  if (!resR)
    return;

  int absVal = std::abs(resR);
  _aec->encode(--absVal > 0, _ctxResRGTOne[interCtx][ctxLR]);
  if (absVal)
    _aec->encode(--absVal > 0, _ctxResRGTTwo[interCtx][ctxLR]);
  if (absVal)
    _aec->encodeExpGolomb(
      absVal - 1, 2, _ctxResRExpGolombPre[interCtx][ctxLR],
      _ctxResRExpGolombSuf[interCtx][ctxLR]);

  int ctxR =
    (_precAzimuthStepDelta ? 4 : 0) + (multiplier ? 2 : 0) + _precSignR;

  _aec->encode(
    resR < 0, _ctxResRSign[interCtx ? 2 : _prevInterFlag][ctxL][ctxR]);
  _precSignR = resR < 0;
  _precAzimuthStepDelta = multiplier;
  _prevInterFlag = interFlag;
}

//-------------------------------------------------------------------------

void
PredGeomEncoder::encodeResPhi(
  int32_t resPhi,
  int predIdx,
  int boundPhi,
  const bool interFlag,
  int refNodeIdx)
{
  int interCtxIdx = interFlag ? 1 : 0;
  int ctxL = interFlag ? (refNodeIdx > 1 ? 1 : 0) : (predIdx ? 1 : 0);
  //int ctxL = predIdx ? 1 : 0;

  _aec->encode(resPhi != 0, _ctxResPhiGTZero[interCtxIdx][ctxL]);
  if (!resPhi)
    return;

  int absVal = std::abs(resPhi);
  _aec->encode(--absVal > 0, _ctxResPhiGTOne[interCtxIdx][ctxL]);
  int interEGkCtxIdx = interFlag ? (refNodeIdx > 1 ? 2 : 1) : 0;
  if (absVal)
    _aec->encodeExpGolomb(
      absVal - 1, 1, _ctxResPhiExpGolombPre[interEGkCtxIdx],
      _ctxResPhiExpGolombSuf[interEGkCtxIdx]);

  _aec->encode(
    resPhi < 0, _ctxResPhiSign[ctxL][interCtxIdx ? 4 : _resPhiOldSign]);
  _resPhiOldSign = interFlag ? (refNodeIdx > 1 ? 3 : 2) : (resPhi < 0 ? 1 : 0);
}

//-------------------------------------------------------------------------

template<size_t NumPrefixCtx, size_t NumSuffixCtx>
inline float
PredGeomEncoder::estimateExpGolomb(
  unsigned int symbol,
  int k,
  AdaptiveBitModel (&ctxPrefix)[NumPrefixCtx],
  AdaptiveBitModel (&ctxSuffix)[NumSuffixCtx])
{
  float bits=0;
  constexpr int maxPrefixIdx = NumPrefixCtx - 1;
  constexpr int maxSuffixIdx = NumSuffixCtx - 1;
  const int k0 = k;

  while (symbol >= (1u << k)) {
    bits += estimate(1, ctxPrefix[std::min(maxPrefixIdx, k - k0)]);
    symbol -= 1u << k;
    k++;
  }
  bits += estimate(0, ctxPrefix[std::min(maxPrefixIdx, k - k0)]);

  while (k--)
    bits += estimate((symbol >> k) & 1, ctxSuffix[std::min(maxSuffixIdx, k)]);

  return bits;
}

float
PredGeomEncoder::estimateResPhi(
  int32_t resPhi,
  int predIdx,
  int boundPhi,
  const bool interFlag,
  int refNodeIdx)
{
  float bits = 0.;
  int interCtxIdx = interFlag ? 1 : 0;

  int ctxL = interFlag ? (refNodeIdx > 1 ? 1 : 0) : (predIdx ? 1 : 0);
  //int ctxL = predIdx ? 1 : 0;

  bits += estimate(resPhi != 0, _ctxResPhiGTZero[interCtxIdx][ctxL]);
  if (!resPhi)
    return bits;

  int absVal = std::abs(resPhi);
  bits += estimate(--absVal > 0, _ctxResPhiGTOne[interCtxIdx][ctxL]);
  int interEGkCtxIdx = interFlag ? (refNodeIdx > 1 ? 2 : 1) : 0;
  if (absVal) 
  {
    absVal = absVal - 1;
    bits += std::max(2, (ilog2(uint32_t(absVal + 2)) << 1));
  }
  //     bits += estimateExpGolomb(
  //       absVal - 1, 1, _ctxResPhiExpGolombPre[interEGkCtxIdx],
  //       _ctxResPhiExpGolombSuf[interEGkCtxIdx]);

  bits += estimate(
    resPhi < 0, _ctxResPhiSign[ctxL][interCtxIdx ? 4 : _resPhiOldSign]);

  return bits;
}

//----------------------------------------------------------------------------
float
PredGeomEncoder::estimateResR(int32_t resR, int multiplier, int predIdx, const bool interFlag
  , int refNodeIdx
)
{
  const int interCtx = interFlag ? 1 : 0;
  float bits = 0.;
  int ctxL = interFlag ? (refNodeIdx > 1 ? 1 : 0)
      : (predIdx ? 1 : 0);
  //int ctxL = predIdx == 0 /* parent */;
  int ctxLR = ctxL + (interFlag ? (abs(multiplier) > 2 ? 2 : 0)
    : (abs(multiplier) > _thQphi ? 2 : 0));
  //int ctxLR = ctxL + (multiplier ? 2 : 0);

  bits += estimate(resR != 0, _ctxResRGTZero[interCtx][ctxLR]);
  if (!resR)
    return bits;

  int absVal = std::abs(resR);
  bits += estimate(--absVal > 0, _ctxResRGTOne[interCtx][ctxLR]);
  if (absVal)
    bits += estimate(--absVal > 0, _ctxResRGTTwo[interCtx][ctxLR]);
  if (absVal) {
    // encode residual by expGolomb k=2
    absVal--;
    bits += std::max(3, (ilog2(uint32_t(absVal + 4)) << 1) - 1);
  }

  int ctxR =
    (_precAzimuthStepDelta ? 4 : 0) + (multiplier ? 2 : 0) + _precSignR;

  bits += estimate(
    resR < 0, _ctxResRSign[interCtx ? 2 : _prevInterFlag][ctxL][ctxR]);
  return bits;
}

//----------------------------------------------------------------------------
void
PredGeomEncoder::encodeResidual(const Vec3<int32_t>& residual, int iMode, int multiplier, int rPred, int predIdx, const bool interFlag
  , int refNodeIdx
)
{
  int interCtxIdx = interFlag ? 1 : 0;
  int k = 0;

  if (_azimuth_scaling_enabled_flag) {
    // N.B. mode is always 1 with _azimuth_scaling_enabled_flag
    encodeResR(residual[0], multiplier, predIdx, interFlag
      , refNodeIdx
    );

    int r = rPred + residual[0] << 3;
    auto speedTimesR = int64_t(_geomAngularAzimuthSpeed) * r;
    int phiBound = divExp2RoundHalfInf(speedTimesR, _azimuthTwoPiLog2 + 1);
    encodeResPhi(residual[1], predIdx, phiBound, interFlag
      , refNodeIdx
    );
    k = 2;
  }

  for (int ctxIdx = 0; k < 3; k++) {
    // The last component (delta laseridx) isn't coded if there is one laser
    if (_geom_angular_mode_enabled_flag && _numLasers == 1 && k == 2)
      continue;

    const auto res = residual[k];
    _aec->encode(res != 0, _ctxResGt0[interCtxIdx][k]);
    if (!res)
      continue;

    int32_t value = abs(res) - 1;
    int32_t numBits = 1 + ilog2(uint32_t(value));

    AdaptiveBitModel* ctxs = &_ctxNumBits[interCtxIdx][ctxIdx][k][0] - 1;
    for (int ctxIdx = 1, n = _pgeom_resid_abs_log2_bits[k] - 1; n >= 0; n--) {
      auto bin = (numBits >> n) & 1;
      _aec->encode(bin, ctxs[ctxIdx]);
      ctxIdx = (ctxIdx << 1) | bin;
    }

    if (!k && !_geom_angular_mode_enabled_flag)
      ctxIdx = std::min(4, (numBits + 1) >> 1);

    --numBits;
    for (int32_t i = 0; i < numBits; ++i)
      _aec->encode((value >> i) & 1);

    if (iMode || k) {
      _aec->encode(res < 0, _ctxSign[interCtxIdx][k]);
    }
  }
}


//----------------------------------------------------------------------------

void
PredGeomEncoder::encodeResidual2(const Vec3<int32_t>& residual)
{
  for (int k = 0; k < 3; k++) {
    const auto res = residual[k];
    _aec->encode(res != 0, _ctxResidual2GtN[0][k]);
    if (!res)
      continue;

    int value = abs(res) - 1;
    _aec->encode(value > 0, _ctxResidual2GtN[1][k]);
    if (value)
      _aec->encodeExpGolomb(value - 1, 0, _ctxEG2Prefix[k], _ctxEG2Suffix[k]);

    _aec->encode(res < 0, _ctxSign2[k]);
  }
}

//----------------------------------------------------------------------------

void
PredGeomEncoder::encodePhiMultiplier(int32_t multiplier, const bool interFlag
  , int refNodeIdx
  , int predIdx
)
{
  int ctxL = interFlag ? (refNodeIdx > 1 ? 1 : 0)
    : (predIdx ? 1 : 0);
  int interCtxIdx = interFlag ? 1 : 0;
  _aec->encode(multiplier != 0, _ctxPhiGtN[interCtxIdx][ctxL][0]);
  //_aec->encode(multiplier != 0, _ctxPhiGtN[interCtxIdx][0]);
  if (!multiplier)
    return;

  int32_t value = abs(multiplier) - 1;
  _aec->encode(value > 0, _ctxPhiGtN[interCtxIdx][ctxL][1]);
  //_aec->encode(value > 0, _ctxPhiGtN[interCtxIdx][1]);
  if (!value) {
    _aec->encode(multiplier < 0, _ctxSignPhi[interCtxIdx][ctxL]);
    //_aec->encode(multiplier < 0, _ctxSignPhi[interCtxIdx]);
    return;
  }

  value--;
  int valueMinus7 = value - 7;
  value = std::min(value, 7);
  _aec->encode((value >> 2) & 1, _ctxResidualPhi[interCtxIdx][ctxL][0]);
  _aec->encode(
    (value >> 1) & 1, _ctxResidualPhi[interCtxIdx][ctxL][1 + (value >> 2)]);
  _aec->encode(
    (value >> 0) & 1, _ctxResidualPhi[interCtxIdx][ctxL][3 + (value >> 1)]);

  if (valueMinus7 >= 0)
    _aec->encodeExpGolomb(valueMinus7, 0, _ctxEGPhi[interCtxIdx][ctxL]);

  _aec->encode(multiplier < 0, _ctxSignPhi[interCtxIdx][ctxL]);
}
//----------------------------------------------------------------------------
void
PredGeomEncoder::encodeInterFlag(bool interFlag, const uint8_t interFlagBuffer)
{
  uint8_t interFlagCtxIdx =
    interFlagBuffer & PredGeomEncoder::interFlagBufferMask;
  _aec->encode(interFlag, _ctxInterFlag[interFlagCtxIdx]);
}

//----------------------------------------------------------------------------

void
PredGeomEncoder::encodeRefDirFlag(bool refDirFlag)
{
  _aec->encode(refDirFlag, _ctxRefDirFlag);
}

//----------------------------------------------------------------------------

void
PredGeomEncoder::encodeRefNodeIdx(int refNodeIdx, bool globalMotionEnabled)
{
  if (globalMotionEnabled)
    _aec->encode((refNodeIdx >> 1) & 1, _ctxRefNodeIdx[0]);
  _aec->encode(refNodeIdx & 1, _ctxRefNodeIdx[1 + (refNodeIdx >> 1)]);
}

//----------------------------------------------------------------------------

void
PredGeomEncoder::encodeQpOffset(int dqp)
{
  _aec->encode(dqp != 0, _ctxQpOffsetAbsGt0);
  if (dqp == 0)
    return;

  _aec->encode(dqp < 0, _ctxQpOffsetSign);
  _aec->encodeExpGolomb(abs(dqp) - 1, 0, _ctxQpOffsetAbsEgl);
}

//----------------------------------------------------------------------------

void
PredGeomEncoder::encodeEndOfTreesFlag(int end_of_trees_flag)
{
  _aec->encode(end_of_trees_flag, _ctxEndOfTrees);
}

//----------------------------------------------------------------------------

float
PredGeomEncoder::estimateBits(
  GPredicter::Mode mode,
  int predIdx,
  const Vec3<int32_t>& residual,
  int multiplier,
  int rPred,
  bool interFlag,
  bool interEnabledFlag,
  int refNodeIdx, bool globalMotionEnabled,
  int numRef,
  bool refDirFlag,
  //bool refNodeFlag,
  const uint8_t interFlagBuffer,
  float best_known_bits)
{
  int iMode = int(mode);
  float bits = 0.;
  int interCtxIdx = interFlag ? 1 : 0;
  if (!interFlag) {
    if (_azimuth_scaling_enabled_flag) {
      for (int i = 0; i < predIdx; ++i)
        bits += estimate(1, _ctxPredIdx[i]);
      if (predIdx < _maxPredIdx)
        bits += estimate(0, _ctxPredIdx[predIdx]);
    } else {
      bits += estimate((iMode >> 1) & 1, _ctxPredMode[0]);
      bits += estimate(iMode & 1, _ctxPredMode[1 + (iMode >> 1)]);
    }
  } else {
    if (numRef > 1)
      bits += estimate(refDirFlag, _ctxRefDirFlag);
    if (globalMotionEnabled)
      bits += estimate((refNodeIdx >> 1) & 1, _ctxRefNodeIdx[0]);
    bits += estimate(refNodeIdx & 1, _ctxRefNodeIdx[1 + (refNodeIdx >> 1)]);
  }
  if (bits > best_known_bits)  return bits;

  if (interEnabledFlag) {
    uint8_t interFlagCtxIdx =
      interFlagBuffer & PredGeomEncoder::interFlagBufferMask;
    bits += estimate(interFlag, _ctxInterFlag[interFlagCtxIdx]);
    if (bits > best_known_bits)  return bits;
  }

  if (_geom_angular_mode_enabled_flag) {
    int ctxL = interFlag ? (refNodeIdx > 1 ? 1 : 0)
      : (predIdx ? 1 : 0);
    interCtxIdx = interFlag ? 1 : 0;
    bits += estimate(multiplier != 0, _ctxPhiGtN[interCtxIdx][ctxL][0]);
    //bits += estimate(multiplier != 0, _ctxPhiGtN[interCtxIdx][0]);
    if (bits > best_known_bits)  return bits;

    if (multiplier) {
      int32_t value = abs(multiplier) - 1;

      bits += estimate(value > 0, _ctxPhiGtN[interCtxIdx][ctxL][1]);
      //bits += estimate(value > 0, _ctxPhiGtN[interCtxIdx][1]);
      bits += estimate(multiplier < 0, _ctxSignPhi[interCtxIdx][ctxL]);
      //bits += estimate(multiplier < 0, _ctxSignPhi[interCtxIdx]);
      if (bits > best_known_bits)  return bits;
      if (value) {
        value--;

        int valueMinus7 = value - 7;
        value = std::min(value, 7);
        bits += estimate((value >> 2) & 1, _ctxResidualPhi[interCtxIdx][ctxL][0]);
        bits += estimate(
          (value >> 1) & 1, _ctxResidualPhi[interCtxIdx][ctxL][1 + (value >> 2)]);
        bits += estimate(
          (value >> 0) & 1, _ctxResidualPhi[interCtxIdx][ctxL][3 + (value >> 1)]);

        if (valueMinus7 >= 0)
          bits += (1 + 2.0 * log2(valueMinus7 + 1));
        if (bits > best_known_bits)  return bits;
      }
    }
  }

  int k = 0;

  if (_azimuth_scaling_enabled_flag) {

    bits += estimateResR(residual[0], multiplier, predIdx, interFlag
      , refNodeIdx
    );
    if (bits > best_known_bits)  return bits;

    int r = rPred + residual[0] << 3;
    auto speedTimesR = int64_t(_geomAngularAzimuthSpeed) * r;
    int phiBound = divExp2RoundHalfInf(speedTimesR, _azimuthTwoPiLog2 + 1);
    bits += estimateResPhi(residual[1], predIdx, phiBound, interFlag
      , refNodeIdx
    );
    if (bits > best_known_bits)  return bits;

    k = 2;
  }

  for (int ctxIdx = 0; k < 3; k++) {
    // The last component (delta laseridx) isn't coded if there is one laser
    if (_geom_angular_mode_enabled_flag && _numLasers == 1 && k == 2)
      continue;

    const auto res = residual[k];
    bits += estimate(res != 0, _ctxResGt0[interCtxIdx][k]);
    if (bits > best_known_bits)  return bits;
    if (res == 0)
      continue;

    if (iMode > 0 || k) {
      bits += estimate(res < 0, _ctxSign[interCtxIdx][k]);
      if (bits > best_known_bits)  return bits;

    }

    int32_t value = abs(res) - 1;
    int32_t numBits = 1 + ilog2(uint32_t(value));

    AdaptiveBitModel* ctxs = &_ctxNumBits[interCtxIdx][ctxIdx][k][0] - 1;
    for (int ctxIdx = 1, n = _pgeom_resid_abs_log2_bits[k] - 1; n >= 0; n--) {
      auto bin = (numBits >> n) & 1;
      bits += estimate(bin, ctxs[ctxIdx]);
      if (bits > best_known_bits)  return bits;
      ctxIdx = (ctxIdx << 1) | bin;
    }

    if (!k && !_geom_angular_mode_enabled_flag)
      ctxIdx = std::min(4, (numBits + 1) >> 1);

    bits += std::max(0, numBits - 1);
    if (bits > best_known_bits)  return bits;
  }
  return bits;
}

//----------------------------------------------------------------------------

int
PredGeomEncoder::encodeTree(
  Vec3<int32_t>* srcPts,
  Vec3<int32_t>* reconPts,
  const GNode* nodes,
  int numNodes,
  int rootIdx,
  int* codedOrder,
  PredGeomPredictor& refFrameSph,
  PredGeomPredictor& refFrameSph2)
{
  QuantizerGeom quantizer(_sliceQp);
  int nodesUntilQpOffset = 0;
  int processedNodes = 0;
  int prevNodeIdx = -1;
  int nodeCount = 0;
  uint8_t interFlagBuffer = 0;

  _stack.push_back(rootIdx);

  const int MaxNPred = kPTEMaxPredictorIndex + 1;
  const int NPred = _maxPredIdx + 1;
  const int NTestedPred = _maxPredIdxTested + 1;

  std::array<std::array<int, 2>, MaxNPred> preds = {};
  const bool frameMovingState =
    refFrameSph.isInterEnabled() && refFrameSph.getFrameMovingState();

  const bool frameMovingState2 = 
    refFrameSph2.isInterEnabled() && refFrameSph2.getFrameMovingState();

  while (!_stack.empty()) {
    const auto nodeIdx = _stack.back();
    _stack.pop_back();
    auto curNodeIdx = nodeCount++;

    const auto& node = nodes[nodeIdx];
    const auto& point = srcPts[nodeIdx];
    struct {
      float bits = bits = std::numeric_limits<float>::max();
      GPredicter::Mode mode;
      int predIdx;
      Vec3<int32_t> residual;
      Vec3<int32_t> prediction;
      int qphi;
      bool interFlag;
      bool RefDirFlag;
      int refNodeIdx = 0;
      //bool refNodeFlag = false;
      uint8_t interFlagBuffer;
    } best;

    if (_geom_scaling_enabled_flag && !nodesUntilQpOffset--) {
      int qp = qpSelector(node);
      quantizer = QuantizerGeom(qp);
      encodeQpOffset((qp - _sliceQp) >> _geom_qp_multiplier_log2);
      nodesUntilQpOffset = _qpOffsetInterval;
    }

    bool isInterEnabled = refFrameSph.isInterEnabled() && (prevNodeIdx >= 0);
    bool isInterEnabled2 = refFrameSph2.isInterEnabled() && (prevNodeIdx >= 0);
    int numRef = isInterEnabled + isInterEnabled2;
    // mode decision to pick best prediction from available set
    int qphi = 0;
    auto azimuthSpeed = _geomAngularAzimuthSpeed;
    std::bitset<4> unusable;

    const int iModeBegin = _azimuth_scaling_enabled_flag ? 1 : 0;
    const int iModeEnd = _azimuth_scaling_enabled_flag ? 2 : 4;
    const int predIdxEnd = _azimuth_scaling_enabled_flag ? NTestedPred : 1;
    bool firstCheck = true;

    for (int iMode = iModeBegin; iMode < iModeEnd; iMode++) {
      for (int predIdx = 0; predIdx < predIdxEnd; ++predIdx) {
        GPredicter::Mode mode = GPredicter::Mode(iMode);
        GPredicter predicter =
          makePredicter(nodeIdx, mode, _pgeom_min_radius, [=](int idx) {
            return nodes[idx].parent;
          });

        if (!_azimuth_scaling_enabled_flag && !predicter.isValid(mode))
          continue;

        bool refDirFlag = 0;
        for (auto interFlag = 0; interFlag
             < (numRef * ((refFrameSph.getGlobalMotionEnabled()) ? 4 : 2)
                               + 1);
             //for (auto interFlag = 0; interFlag < (isInterEnabled ? 3 : 1);
             interFlag++) {
          point_t pred = 0;
          int refNodeIdx = 0;
          //bool refNodeFlag = false;
          if (!interFlag) {
            pred = predicter.predict(
              &srcPts[0], mode, _geom_angular_mode_enabled_flag);

            if (_azimuth_scaling_enabled_flag && predIdx > 0) {
              pred[0] = preds[predIdx][0];
              auto deltaPhi = pred[1] - preds[predIdx][1];
              pred[1] = preds[predIdx][1];
              if (
                deltaPhi >= _geomAngularAzimuthSpeed
                || deltaPhi <= -_geomAngularAzimuthSpeed) {
                int qphi0 =
                  divApprox(int64_t(deltaPhi), _geomAngularAzimuthSpeed, 0);
                pred[1] += qphi0 * _geomAngularAzimuthSpeed;
              }
            }
          } else {
            if (_azimuth_scaling_enabled_flag ? predIdx : iMode)
              continue;
            const auto prevPos = srcPts[prevNodeIdx];
            const auto parentPos = srcPts[nodes[nodeIdx].parent];
            refNodeIdx = interFlag - 1;
            if(numRef > 1){
              if(refFrameSph.getGlobalMotionEnabled()){
                if(refNodeIdx > 3){
                  refNodeIdx -= 4;
                  refDirFlag = 1;
                }
              } 
              else{
                if(refNodeIdx > 1){
                  refNodeIdx -= 2;
                  refDirFlag = 1;
                }
              } 
            }
            const auto interPred = refDirFlag ? refFrameSph2.getInterPred(prevPos[1], prevPos[2], refNodeIdx) :
              refFrameSph.getInterPred(prevPos[1], prevPos[2], refNodeIdx);
            const bool frameMoving = refDirFlag ? frameMovingState2 : frameMovingState;

            if (interPred.first) {
              pred = interPred.second;
              if (refNodeIdx > 1 && frameMoving) {
                const auto deltaPhi = pred[1] - parentPos[1];
                pred[1] = parentPos[1];
                if (
                  deltaPhi >= (_geomAngularAzimuthSpeed >> 1)
                  || deltaPhi <= -(_geomAngularAzimuthSpeed >> 1)) {
                  int qphi0 = divApprox(
                    int64_t(deltaPhi) + (_geomAngularAzimuthSpeed >> 1),
                    _geomAngularAzimuthSpeed, 0);
                  pred[1] += qphi0 * _geomAngularAzimuthSpeed;
                }
              }
            } 
            else
              continue;
          }
          auto residual = point - pred;
          if (!_geom_angular_mode_enabled_flag)
            for (int k = 0; k < 3; k++)
              residual[k] = int32_t(quantizer.quantize(residual[k]));
          else {
            while (residual[1] < -1 << (_azimuthTwoPiLog2 - 1))
              residual[1] += (1 << _azimuthTwoPiLog2);
            while (residual[1] >= 1 << (_azimuthTwoPiLog2 - 1))
              residual[1] -= (1 << _azimuthTwoPiLog2);

            if (_azimuth_scaling_enabled_flag) {
              // Quantize phi by 8r/2^n
              auto r = (pred[0] + residual[0]) << 3;

              azimuthSpeed = _geomAngularAzimuthSpeed;
              qphi = 0;
              auto speedTimesR = int64_t(azimuthSpeed) * r;
              int phiBound =
                divExp2RoundHalfInf(speedTimesR, _azimuthTwoPiLog2 + 1);
              if (r) {
                if (!phiBound) {
                  const int32_t pi = 1 << (_azimuthTwoPiLog2 - 1);
                  int32_t speedTimesR32 = speedTimesR;
                  while (speedTimesR32 < pi) {
                    speedTimesR32 <<= 1;
                    azimuthSpeed <<= 1;
                  }
                }
                qphi = residual[1] >= 0
                  ? (residual[1] + (azimuthSpeed >> 1)) / azimuthSpeed
                  : -(-residual[1] + (azimuthSpeed >> 1)) / azimuthSpeed;
                pred[1] += qphi * azimuthSpeed;
                residual[1] = point[1] - pred[1];
                while (residual[1] < -1 << (_azimuthTwoPiLog2 - 1))
                  residual[1] += (1 << _azimuthTwoPiLog2);
                while (residual[1] >= 1 << (_azimuthTwoPiLog2 - 1))
                  residual[1] -= (1 << _azimuthTwoPiLog2);
              }

              auto arc = int64_t(residual[1]) * r;
              residual[1] =
                int32_t(divExp2RoundHalfInf(arc, _azimuthTwoPiLog2));
              if (residual[1] < -phiBound)
                residual[1] = -phiBound;
              if (residual[1] > phiBound)
                residual[1] = phiBound;
              // lossless encoding of theta index
            } else {
              // The residual in the spherical domain is losslessly coded
              qphi = residual[1] >= 0
                ? (residual[1] + (_geomAngularAzimuthSpeed >> 1))
                  / _geomAngularAzimuthSpeed
                : -(-residual[1] + (_geomAngularAzimuthSpeed >> 1))
                  / _geomAngularAzimuthSpeed;
              pred[1] += qphi * _geomAngularAzimuthSpeed;
              residual[1] = point[1] - pred[1];
            }
          }

          // Check if the prediction residual can be represented with the
          // current configuration.  If it can't, don't use this mode.
          for (int k = 0; k < 3; k++) {
            if (residual[k])
              if ((abs(residual[k]) - 1) >> _maxAbsResidualMinus1Log2[k])
                unusable[iMode] = true;
          }

          if (unusable[iMode]) {
            if (iMode == 3 && unusable.all())
              throw std::runtime_error(
                "predgeom: can't represent residual in any mode");
            if (iMode > 0)
              continue;
          }

          // penalize bit cost so that it is only used if all modes overfow
          auto bits = estimateBits(
            mode, predIdx, residual, qphi, pred[0], interFlag, isInterEnabled,
            refNodeIdx,
            refFrameSph.getGlobalMotionEnabled(),
            numRef,
            refDirFlag,
            //refNodeFlag,
            interFlagBuffer, best.bits);

          if (unusable[iMode])
            bits = std::numeric_limits<decltype(bits)>::max();

          if (firstCheck || bits < best.bits) {
            best.prediction = pred;
            best.predIdx = predIdx;
            best.residual = residual;
            best.mode = mode;
            best.bits = bits;
            best.qphi = qphi;
            best.interFlag = interFlag;
            firstCheck = false;
            best.refNodeIdx = refNodeIdx;
            best.RefDirFlag = refDirFlag;
            //best.refNodeFlag = refNodeFlag;
          }
        }
      }
    }

    assert(node.childrenCount <= GNode::MaxChildrenCount);
    if (!_geom_unique_points_flag)
      encodeNumDuplicatePoints(node.numDups);
    encodeNumChildren(node.childrenCount);
    if (isInterEnabled)
      encodeInterFlag(best.interFlag, interFlagBuffer);
    if (best.interFlag){
      if (numRef > 1)
        encodeRefDirFlag(best.RefDirFlag);
      encodeRefNodeIdx(best.refNodeIdx, refFrameSph.getGlobalMotionEnabled());      
    }

      //encodeRefNodeFlag(best.refNodeFlag);
    else {
      if (_azimuth_scaling_enabled_flag)
        encodePredIdx(best.predIdx);
      else
        encodePredMode(best.mode);
    }

    if (_geom_angular_mode_enabled_flag)
      encodePhiMultiplier(best.qphi, best.interFlag
        , best.refNodeIdx
        , best.predIdx
      );

    encodeResidual(
      best.residual, best.mode, best.qphi, best.prediction[0], best.predIdx,
      best.interFlag
      , best.refNodeIdx
    );

    // convert spherical prediction to cartesian and re-calculate residual
    if (_geom_angular_mode_enabled_flag) {
      if (_azimuth_scaling_enabled_flag) {
        auto r = (best.prediction[0] + best.residual[0]) << 3;
        if (!r)
          r = 1;

        // scale the azimuth residual
        int32_t rInvScaleLog2;
        int64_t rInv = recipApprox(r, rInvScaleLog2);
        best.residual[1] =
          divExp2(best.residual[1] * rInv, rInvScaleLog2 - _azimuthTwoPiLog2);

        // update original spherical position with reconstructed spherical
        // position, for use in, for instance, attribute coding.
        srcPts[nodeIdx] /* == point */ = best.prediction + best.residual;
        if (srcPts[nodeIdx][1] < -1 << (_azimuthTwoPiLog2 - 1))
          srcPts[nodeIdx][1] += (1 << _azimuthTwoPiLog2);
        if (srcPts[nodeIdx][1] >= 1 << (_azimuthTwoPiLog2 - 1))
          srcPts[nodeIdx][1] -= (1 << _azimuthTwoPiLog2);
        for (int i = 1; i <= node.numDups; i++)
          srcPts[nodeIdx + i] = srcPts[nodeIdx];

        bool flagNewObject = (best.interFlag ? std::abs(point[0] - preds[0][0])
                                             : std::abs(best.residual[0]))
          > _thObj;

        int predIdx = flagNewObject ? NPred - 1 : best.predIdx;
        for (int i = predIdx; i > 0; i--)
          preds[i] = preds[i - 1];
        preds[0][0] = srcPts[nodeIdx][0];
        preds[0][1] = srcPts[nodeIdx][1];
      }

      best.prediction = origin + _sphToCartesian(point);
      best.residual = reconPts[nodeIdx] - best.prediction;
      for (int k = 0; k < 3; k++)
        best.residual[k] = int32_t(quantizer.quantize(best.residual[k]));
      if (!_predgeometry_residual2_disabling_enabled_flag) {
        encodeResidual2(best.residual);
      } else {
        best.residual = 0;
      }
    }

    // write the reconstructed position back to the point cloud
    for (int k = 0; k < 3; k++)
      best.residual[k] = int32_t(quantizer.scale(best.residual[k]));
    reconPts[nodeIdx] = best.prediction + best.residual;

    for (int k = 0; k < 3; k++)
      reconPts[nodeIdx][k] = std::max(0, reconPts[nodeIdx][k]);

    // NB: the coded order of duplicate points assumes that the duplicates
    // are consecutive -- in order that the correct attributes are coded.
    codedOrder[processedNodes++] = nodeIdx;
    for (int i = 1; i <= node.numDups; i++)
      codedOrder[processedNodes++] = nodeIdx + i;

    if (_geom_angular_mode_enabled_flag) {
      // duplicated points must be updated with unquantized value
      // for attributes coding
      for (int i = 1; i <= node.numDups; i++)
        srcPts[nodeIdx + i] = srcPts[nodeIdx];
    }

    for (int i = 0; i < node.childrenCount; i++) {
      _stack.push_back(node.children[i]);
    }

    prevNodeIdx = nodeIdx;
    interFlagBuffer = (interFlagBuffer << 1) | (best.interFlag ? 1 : 0);
  }

  return processedNodes;
}

//----------------------------------------------------------------------------

void
PredGeomEncoder::encode(
  Vec3<int32_t>* cloudA,
  Vec3<int32_t>* cloudB,
  const GNode* nodes,
  int numNodes,
  int32_t* codedOrder,
  PredGeomPredictor& refFrameSph,
  PredGeomPredictor& refFrameSph2, 
  bool reversed)
{
  int32_t processedNodes = 0;
  int32_t rootIdx = reversed ? numNodes-1 : 0;   
  int32_t step = reversed ? -1 : +1;   
  for (; (rootIdx >= 0 && rootIdx < numNodes) ; rootIdx +=step) {
    // find the root node(s)
    if (nodes[rootIdx].parent >= 0)
      continue;

    int numSubtreeNodes = encodeTree(
      cloudA, cloudB, nodes, numNodes, rootIdx, codedOrder + processedNodes,
      refFrameSph, refFrameSph2);
    processedNodes += numSubtreeNodes;

    // NB: this is just in case this call to encode needs to encode an
    // additional tree.  The actual end_of_trees_flag = true is signalled
    // at the end.
    if (processedNodes != numNodes)
      encodeEndOfTreesFlag(false);
  }
  assert(processedNodes == numNodes);
}

//============================================================================

std::vector<GNode>
generateGeomPredictionTree(
  const GeometryParameterSet& gps,
  const Vec3<int32_t>* begin,
  const Vec3<int32_t>* end)
{
  const int32_t pointCount = std::distance(begin, end);

  // Build the prediction tree, roughly as follows:
  //  - For each point, find the node with the nearest prediction
  //    with empty children slots:
  //     - generate new predictions based upon the current point and
  //       insert into the search tree
  using NanoflannKdTreeT = nanoflann::KDTreeSingleIndexDynamicAdaptor<
    nanoflann::L2_Simple_Adaptor<int32_t, NanoflannCloud, int64_t>,
    NanoflannCloud, 3, int32_t>;

  using NanoflannResultT = nanoflann::KNNResultSet<int64_t, int32_t>;

  // the predicted point positions, used for searching.
  // each node will generate up to three predicted positions
  NanoflannCloud predictedPoints;
  NanoflannKdTreeT predictedPointsTree(3, predictedPoints);
  predictedPoints.pts.reserve(3 * pointCount);

  // mapping of predictedPoints indicies to prediction tree nodes
  std::vector<int32_t> predictedPointIdxToNodeIdx;
  predictedPointIdxToNodeIdx.reserve(3 * pointCount);

  // the prediction tree, one node for each point
  std::vector<GNode> nodes(pointCount);

  for (int nodeIdx = 0, nodeIdxN; nodeIdx < pointCount; nodeIdx = nodeIdxN) {
    auto& node = nodes[nodeIdx];
    auto queryPoint = begin[nodeIdx];

    // scan for duplicate points
    // NB: the tree coder assumes that duplicate point indices are consecutive
    // If this guarantee is changed, then the tree coder must be modified too.
    node.numDups = 0;
    for (nodeIdxN = nodeIdx + 1; nodeIdxN < pointCount; nodeIdxN++) {
      if (queryPoint != begin[nodeIdxN])
        break;
      node.numDups++;
    }

    int32_t nnPredictedPointIdx[GNode::MaxChildrenCount];
    int64_t nnPredictedDist[GNode::MaxChildrenCount];
    NanoflannResultT resultSet(GNode::MaxChildrenCount);

    resultSet.init(nnPredictedPointIdx, nnPredictedDist);
    predictedPointsTree.findNeighbors(resultSet, &queryPoint[0], {});

    // find a suitable parent.  default case: node has no parent
    node.parent = -1;
    node.childrenCount = 0;
    for (size_t r = 0; r < resultSet.size(); ++r) {
      auto parentIdx = predictedPointIdxToNodeIdx[nnPredictedPointIdx[r]];
      auto& pnode = nodes[parentIdx];
      if (pnode.childrenCount < GNode::MaxChildrenCount) {
        node.parent = parentIdx;
        pnode.children[pnode.childrenCount++] = nodeIdx;
        break;
      }
    }

    // update the search tree with new predictions from this point
    const auto size0 = predictedPoints.pts.size();

    // set the indicies for prediction
    GPredicter predicter;
    predicter.index[0] = nodeIdx;
    predicter.index[1] = nodes[predicter.index[0]].parent;
    if (predicter.index[1] >= 0)
      predicter.index[2] = nodes[predicter.index[1]].parent;

    for (int iMode = 0; iMode < 4; iMode++) {
      GPredicter::Mode mode = GPredicter::Mode(iMode);

      // don't add zero prediction
      if (mode == GPredicter::None)
        continue;

      if (!predicter.isValid(mode))
        continue;

      auto prediction = predicter.predict(begin, mode, false);
      predictedPointIdxToNodeIdx.push_back(nodeIdx);
      predictedPoints.pts.push_back(prediction);
    }

    const auto size1 = predictedPoints.pts.size();
    if (size0 != size1)
      predictedPointsTree.addPoints(size0, size1 - 1);
  }

  return nodes;
}

//----------------------------------------------------------------------------

std::vector<GNode>
generateGeomPredictionTreeAngular(
  const GeometryParameterSet& gps,
  const Vec3<int32_t> origin,
  const Vec3<int32_t>* begin,
  const Vec3<int32_t>* end,
  Vec3<int32_t>* beginSph,
  bool enablePartition,
  int32_t splitter,
  bool& reversed)
{
  int32_t pointCount = std::distance(begin, end);
  int32_t numLasers = gps.numLasers();

  // the prediction tree, one node for each point
  std::vector<GNode> nodes(pointCount);
  std::vector<int32_t> prevNodes(numLasers, -1);
  std::vector<int32_t> firstNodes(numLasers, -1);
  std::vector<int32_t> prevNodesObj(numLasers, -1);
  std::vector<int32_t> firstNodesObj(numLasers, -1);

  bool isRoad=false;
  CartesianToSpherical cartToSpherical(gps);

  for (int nodeIdx = 0, nodeIdxN; nodeIdx < pointCount; nodeIdx = nodeIdxN) {
    auto curPoint = begin[nodeIdx];
    isRoad = enablePartition ? (curPoint[2] > splitter ? false : true) : true;

    auto& node = nodes[nodeIdx];
    node.childrenCount = 0;

    // scan for duplicate points
    // NB: the tree coder assumes that duplicate point indices are consecutive
    // If this guarantee is changed, then the tree coder must be modified too.
    node.numDups = 0;
    for (nodeIdxN = nodeIdx + 1; nodeIdxN < pointCount; nodeIdxN++) {
      if (curPoint != begin[nodeIdxN])
        break;
      node.numDups++;
    }

    // cartesian to spherical coordinates
    const auto carPos = curPoint - origin;
    auto& sphPos = beginSph[nodeIdx] = cartToSpherical(carPos);
    auto thetaIdx = sphPos[2];

    // propagate converted coordinates over duplicate points
    for (int i = nodeIdx + 1; i < nodeIdxN; i++)
      beginSph[i] = sphPos;

    if (isRoad == true) {
      node.parent = prevNodes[thetaIdx];
      if (node.parent != -1) {
        auto& pnode = nodes[prevNodes[thetaIdx]];
        pnode.children[pnode.childrenCount++] = nodeIdx;
      } else
      	firstNodes[thetaIdx] = nodeIdx;
      prevNodes[thetaIdx] = nodeIdx;
    } else {
      node.parent = prevNodesObj[thetaIdx];
      if (node.parent != -1) {
        auto& pnode = nodes[prevNodesObj[thetaIdx]];
        pnode.children[pnode.childrenCount++] = nodeIdx;
      } else
        firstNodesObj[thetaIdx] = nodeIdx;

      prevNodesObj[thetaIdx] = nodeIdx;
    }
  }

  int32_t n0 = 0;
  while (firstNodes[n0] == -1)
    ++n0;
  int32_t roadIdx = firstNodes[n0];

  for (int32_t n = n0 + 1, parentIdx = firstNodes[n0]; n < numLasers; ++n) {
    auto nodeIdx = firstNodes[n];
    if (nodeIdx < 0)
      continue;

    auto& pnode = nodes[parentIdx];
    if (pnode.childrenCount < GNode::MaxChildrenCount) {
      nodes[nodeIdx].parent = parentIdx;
      pnode.children[pnode.childrenCount++] = nodeIdx;
    }
    parentIdx = nodeIdx;
  }

  if (enablePartition) {
    n0 = 0;
    while (firstNodesObj[n0] == -1)
      ++n0;
    int32_t objIdx = firstNodesObj[n0];
    if (roadIdx > objIdx) reversed =true;

    for (int32_t n = n0 + 1, parentIdx = firstNodesObj[n0]; n < numLasers; ++n) {
      auto nodeIdx = firstNodesObj[n];
      if (nodeIdx < 0)
        continue;

      auto& pnode = nodes[parentIdx];
      if (pnode.childrenCount < GNode::MaxChildrenCount) {
        nodes[nodeIdx].parent = parentIdx;
        pnode.children[pnode.childrenCount++] = nodeIdx;
      }
      parentIdx = nodeIdx;
    }   
  }


  return nodes;
}

//============================================================================

static void
mortonSort(PCCPointSet3& cloud, int begin, int end, int depth)
{
  radixSort8(
    depth, PCCPointSet3::iterator(&cloud, begin),
    PCCPointSet3::iterator(&cloud, end),
    [=](int depth, const PCCPointSet3::Proxy& proxy) {
      const auto& point = *proxy;
      int mask = 1 << depth;
      return !!(point[2] & mask) | (!!(point[1] & mask) << 1)
        | (!!(point[0] & mask) << 2);
    });
}

//============================================================================

Vec3<int>
originFromLaserAngle(const PCCPointSet3& cloud)
{
  Vec3<int> origin = 0;
  auto numPoints = cloud.getPointCount();
  int i;

  for (i = 0; i < numPoints; i++) {
    if (cloud.getLaserAngle(i) == 90) {
      origin = cloud[i];
      break;
    }
  }

  for (; i < numPoints; i++) {
    if (cloud.getLaserAngle(i) == 90) {
      // the most left point
      if (origin[0] > cloud[i][0])
        origin = cloud[i];
    }
  }

  return origin;
}

//============================================================================

void
encodePredictiveGeometry(
  const PredGeomEncOpts& opt,
  const GeometryParameterSet& gps,
  GeometryBrickHeader& gbh,
  PCCPointSet3& cloud,
  std::vector<point_t>* reconPosSph,
  PredGeomPredictor& refFrameSph,
  PredGeomPredictor& refFrameSph2,
  PredGeomContexts& ctxtMem,
  EntropyEncoder* arithmeticEncoder)
{
  auto numPoints = cloud.getPointCount();

  // Origin relative to slice origin
  //  - if angular disabled, try to find a presort origin using laser angles
  Vec3<int> origin = gbh.geomAngularOrigin(gps);
  if (!gps.geom_angular_mode_enabled_flag && cloud.hasLaserAngles())
    origin = originFromLaserAngle(cloud);

  // storage for reordering the output point cloud
  PCCPointSet3 outCloud;
  outCloud.addRemoveAttributes(cloud.hasColors(), cloud.hasReflectances());
  outCloud.resize(numPoints);

  // storage for spherical point co-ordinates determined in angular mode
  std::vector<Vec3<int32_t>> sphericalPos;
  if (!gps.geom_angular_mode_enabled_flag)
    reconPosSph = nullptr;
  else {
    sphericalPos.resize(numPoints);
    if (reconPosSph)
      reconPosSph->resize(numPoints);
  }

  // src indexes in coded order
  std::vector<int32_t> codedOrder(numPoints, -1);

  // Assume that the number of bits required for residuals is equal to the
  // quantised root node size.  This allows every position to be coded with PCM
  if (!gps.geom_angular_mode_enabled_flag) {
    QuantizerGeom quant(gbh.sliceQp(gps));
    for (int k = 0; k < 3; k++) {
      int max = quant.quantize((1 << gbh.rootNodeSizeLog2[k]) - 1);
      gbh.pgeom_resid_abs_log2_bits[k] = numBits(ceillog2(std::max(1, max)));
    }
  }

  // Number of residual bits bits for angular mode.  This is slightly
  // pessimistic in the calculation of r.
  if (gps.geom_angular_mode_enabled_flag) {
    auto xyzBboxLog2 = gbh.rootNodeSizeLog2;
    auto rDivLog2 = gps.geom_angular_radius_inv_scale_log2;

    // first work out the maximum number of bits for the residual
    // NB: the slice coordinate system is used here: ie, minX|minY = 0
    int maxX = (1 << xyzBboxLog2[0]) - 1;
    int maxY = (1 << xyzBboxLog2[1]) - 1;
    int maxAbsDx = std::max(std::abs(origin[0]), std::abs(maxX - origin[0]));
    int maxAbsDy = std::max(std::abs(origin[1]), std::abs(maxY - origin[1]));
    auto r = (int64_t)std::round(std::hypot(maxAbsDx, maxAbsDy));

    Vec3<int> residualBits;
    residualBits[0] = ceillog2(divExp2RoundHalfUp(r, rDivLog2));
    residualBits[2] = ceillog2(gps.numLasers() - 1);
    if (!gps.azimuth_scaling_enabled_flag)
      residualBits[1] =
        ceillog2((gps.geom_angular_azimuth_speed_minus1 + 1) >> 1);
    else {
      int maxError = ((gps.geom_angular_azimuth_speed_minus1 + 1) >> 1) + 1;
      int twoPi = gps.geom_angular_azimuth_scale_log2_minus11 + 12;
      residualBits[1] = ceillog2(divExp2RoundHalfInf(
        maxError * divExp2RoundHalfUp(r << 3, rDivLog2), twoPi));
    }

    // the number of prefix bits required
    for (int k = 0; k < 3; k++)
      gbh.pgeom_resid_abs_log2_bits[k] = ilog2(uint32_t(residualBits[k])) + 1;
  }

  // set the minimum value to zero, if encoder don't use this info.
  gbh.pgeom_min_radius = 0;
  int32_t splitter = opt.splitter;
  bool reversed = false;

  // determine each geometry tree, and encode.  Size of trees is limited
  // by maxPtsPerTree.
  PredGeomEncoder enc(gps, gbh, opt, ctxtMem, arithmeticEncoder);
  int maxPtsPerTree = std::min(opt.maxPtsPerTree, int(numPoints));
  refFrameSph.init(
    gps.interAzimScaleLog2, gps.numLasers(), gps.globalMotionEnabled,
    gps.resamplingEnabled);
  refFrameSph2.init(
    gps.interAzimScaleLog2, gps.numLasers(), gps.globalMotionEnabled,
    gps.resamplingEnabled);

  for (int i = 0; i < numPoints;) {
    int iEnd = std::min(i + maxPtsPerTree, int(numPoints));
    auto* begin = &cloud[i];
    auto* beginSph = sphericalPos.data() + i;
    auto* end = &cloud[0] + iEnd;

    // first, put the points in this tree into a sorted order
    // this can significantly improve the constructed tree
    if (opt.sortMode == PredGeomEncOpts::kSortMorton)
      mortonSort(cloud, i, iEnd, gbh.maxRootNodeDimLog2);
    else if (opt.sortMode == PredGeomEncOpts::kSortAzimuth)
      sortByAzimuth(cloud, i, iEnd, opt.azimuthSortRecipBinWidth, origin);
    else if (opt.sortMode == PredGeomEncOpts::kSortRadius)
      sortByRadius(cloud, i, iEnd, origin);
    else if (opt.sortMode == PredGeomEncOpts::kSortLaserAngle)
      sortByLaserAngle(cloud, i, iEnd, opt.azimuthSortRecipBinWidth, origin);

    // then build and encode the tree
    auto nodes = gps.geom_angular_mode_enabled_flag
      ? generateGeomPredictionTreeAngular(
          gps, origin, begin, end, beginSph, opt.enablePartition,splitter, reversed)
      : generateGeomPredictionTree(gps, begin, end);

    // Determining minimum radius for prediction.
    // NB: this value is per-slice, but if the slice generates multiple
    // trees due to the maxPtsPerTree limit, the radius of all points in
    // cthe slice is not known and the feature disabled.
    if (gps.geom_angular_mode_enabled_flag && numPoints <= maxPtsPerTree) {
      int min = beginSph[i][0];
      for (int j = i + 1; j < iEnd; j++)
        min = std::min(min, beginSph[j][0]);
      gbh.pgeom_min_radius = min;
      enc.setMinRadius(gbh.pgeom_min_radius);
    }

    auto* a = gps.geom_angular_mode_enabled_flag ? beginSph : begin;
    auto* b = begin;

    if (i > 0)
      enc.encodeEndOfTreesFlag(false);
    enc.encode(a, b, nodes.data(), nodes.size(), codedOrder.data() + i, refFrameSph, refFrameSph2, reversed);

    // put points in output cloud in decoded order
    for (auto iBegin = i; i < iEnd; i++) {
      auto srcIdx = iBegin + codedOrder[i];
      assert(srcIdx >= 0);
      outCloud[i] = cloud[srcIdx];
      if (reconPosSph)
        (*reconPosSph)[i] = sphericalPos[srcIdx];
      if (cloud.hasColors())
        outCloud.setColor(i, cloud.getColor(srcIdx));
      if (cloud.hasReflectances())
        outCloud.setReflectance(i, cloud.getReflectance(srcIdx));
    }
  }

  // the slice has finished
  enc.encodeEndOfTreesFlag(true);

  // save the context state for re-use by a future slice if required
  ctxtMem = enc.getCtx();

  swap(cloud, outCloud);
}

// ============================================================================
// Predictive Geometry Coding Cartesian
// ============================================================================
void
PredGeomEncoder::encodeMode(int mode, int group, int rangeClass, bool boundary)
{
  const int g = group;
  const int rc = std::max(0, std::min(kCartesianRangeClasses - 1, rangeClass));
  const int bc = boundary ? 1 : 0;

  for (int i = 0; i < mode; ++i)
    _aec->encode(1, _ctxPredMode_g[g][rc][bc][i]);
  if (mode < kCartesianPredModes - 1)
    _aec->encode(0, _ctxPredMode_g[g][rc][bc][mode]);
}

float
PredGeomEncoder::estimateModeBits(int mode, int group, int rangeClass, bool boundary)
{
  const int g = group;
  const int rc = std::max(0, std::min(kCartesianRangeClasses - 1, rangeClass));
  const int bc = boundary ? 1 : 0;
  float bits = 0.F;

  for (int i = 0; i < mode; ++i)
    bits += estimate(1, _ctxPredMode_g[g][rc][bc][i]);
  if (mode < kCartesianPredModes - 1)
    bits += estimate(0, _ctxPredMode_g[g][rc][bc][mode]);

  return bits;
}

void
PredGeomEncoder::encodeHistIdx(
  int histIdx, int group, int rangeClass, bool boundary)
{
  const int g = group;
  const int rc = std::max(0, std::min(kCartesianRangeClasses - 1, rangeClass));
  const int bc = boundary ? 1 : 0;

  for (int i = 0; i < histIdx; ++i)
    _aec->encode(1, _ctxHistIdx_g[g][rc][bc][i]);
  if (histIdx < kHistSize - 1)
    _aec->encode(0, _ctxHistIdx_g[g][rc][bc][histIdx]);
}

float
PredGeomEncoder::estimateHistIdxBits(
  int histIdx, int group, int rangeClass, bool boundary)
{
  const int g = group;
  const int rc = std::max(0, std::min(kCartesianRangeClasses - 1, rangeClass));
  const int bc = boundary ? 1 : 0;
  float bits = 0.F;

  for (int i = 0; i < histIdx; ++i)
    bits += estimate(1, _ctxHistIdx_g[g][rc][bc][i]);
  if (histIdx < kHistSize - 1)
    bits += estimate(0, _ctxHistIdx_g[g][rc][bc][histIdx]);
  return bits;
}

void
PredGeomEncoder::encodePredGeom(
  const Vec3<int32_t>& residual,
  int group,
  int mode,
  int rangeClass,
  bool boundary)
{
  const int g = group;
  const int rc = std::max(0, std::min(kCartesianRangeClasses - 1, rangeClass));
  const int bc = boundary ? 1 : 0;
  const int mc = std::max(0, std::min(kCartesianPredModes - 1, mode));
  const int pf = predFamily(mc);

  int k = 0;
  for (int ctxIdx = 0; k < 3; k++) {
    const auto res = residual[k];
    _aec->encode(res != 0, _ctxResGt0_g[g][rc][bc][mc][k]);
    if (!res)
      continue;

    int32_t value = abs(res) - 1;
    int32_t numBits = 1 + ilog2(uint32_t(value));

    AdaptiveBitModel* ctxs = &_ctxNumBits_g[g][rc][bc][ctxIdx][k][0] - 1;
    for (int ctxIdx = 1, n = _pgeom_resid_abs_log2_bits[k] - 1; n >= 0; n--) {
      auto bin = (numBits >> n) & 1;
      _aec->encode(bin, ctxs[ctxIdx]);
      ctxIdx = (ctxIdx << 1) | bin;
    }

    ctxIdx = std::min(7, (numBits + 1) >> 1);

    --numBits;
    for (int32_t i = 0; i < numBits; ++i)
      _aec->encode((value >> i) & 1);

    _aec->encode(res < 0, _ctxSign_g[g][rc][bc][pf][k]);
  }
}

float
PredGeomEncoder::estimatePredGeomBits(
  const Vec3<int32_t>& residual,
  int group,
  int mode,
  int rangeClass,
  bool boundary)
{
  const int g = group;
  const int rc = std::max(0, std::min(kCartesianRangeClasses - 1, rangeClass));
  const int bc = boundary ? 1 : 0;
  const int mc = std::max(0, std::min(kCartesianPredModes - 1, mode));
  const int pf = predFamily(mc);

  float bits = 0.F;
  int k = 0;
  for (int ctxIdx = 0; k < 3; k++) {
    const auto res = residual[k];
    bits += estimate(res != 0, _ctxResGt0_g[g][rc][bc][mc][k]);
    if (!res)
      continue;

    int32_t value = abs(res) - 1;
    int32_t numBits = 1 + ilog2(uint32_t(value));

    AdaptiveBitModel* ctxs = &_ctxNumBits_g[g][rc][bc][ctxIdx][k][0] - 1;
    for (int ctxIdx = 1, n = _pgeom_resid_abs_log2_bits[k] - 1; n >= 0; n--) {
      auto bin = (numBits >> n) & 1;
      bits += estimate(bin, ctxs[ctxIdx]);
      ctxIdx = (ctxIdx << 1) | bin;
    }

    ctxIdx = std::min(7, (numBits + 1) >> 1);
    bits += std::max(0, numBits - 1);
    bits += estimate(res < 0, _ctxSign_g[g][rc][bc][pf][k]);
  }

  return bits;
}

// ============================================================================

void
encodePredictiveGeometry(
  const PredGeomEncOpts& opt,
  const GeometryParameterSet& gps,
  GeometryBrickHeader& gbh,
  PCCPointSet3& cloud,
  PredGeomContexts& ctxtMem,
  EntropyEncoder* arithmeticEncoder,
  int numGroups)
{
  auto numPoints = cloud.getPointCount();

  PCCPointSet3 outCloud;
  outCloud.addRemoveAttributes(cloud.hasColors(), cloud.hasReflectances());
  if (cloud.hasLaserAngles())
    outCloud.addLaserAngles();
  outCloud.resize(numPoints);

  std::vector<int32_t> codedOrder(numPoints, -1);

  // Fixed residual bit-depth
  for (int k = 0; k < 3; k++) {
    gbh.pgeom_resid_abs_log2_bits[k] = 5;
  }

  PredGeomEncoder enc(gps, gbh, opt, ctxtMem, arithmeticEncoder, numGroups);

  struct RingState {
    point_t last = 0;
    point_t prev = 0;
    point_t prev2 = 0;
    std::array<point_t, kHistSize> histPts = {};
    int lastR = 0;
    int prevR = 0;
    int prev2R = 0;
    std::array<int, kHistSize> histR = {};
    int count = 0;

    bool hasLast() const { return count >= 1; }
    bool hasPrev() const { return count >= 2; }
    bool hasPrev2() const { return count >= 3; }

    void push(const point_t& pt, int rApprox)
    {
      for (int i = kHistSize - 1; i > 0; --i) {
        histPts[i] = histPts[i - 1];
        histR[i] = histR[i - 1];
      }
      histPts[0] = pt;
      histR[0] = rApprox;
      count = std::min(count + 1, kHistSize);

      last = histPts[0];
      lastR = histR[0];
      prev = count >= 2 ? histPts[1] : point_t(0);
      prevR = count >= 2 ? histR[1] : 0;
      prev2 = count >= 3 ? histPts[2] : point_t(0);
      prev2R = count >= 3 ? histR[2] : 0;
    }
  };

  struct Candidate {
    int mode = kPredZero;
    int histIdx = 0;
    point_t pred = 0;
    int predR = 0;
    int rangeClass = 0;
    bool boundary = false;
    bool valid = true;
  };

  std::array<RingState, kNumRings> reconState = {};

  auto ctxGroupForLaser = [numGroups](int laserIdx) {
    const int ringsPerGroup = std::max(1, kNumRings / numGroups);
    return std::min(numGroups - 1, std::max(0, laserIdx / ringsPerGroup));
  };

  auto roundedDiv = [](int sum, int div) {
    if (sum >= 0)
      return (sum + (div >> 1)) / div;
    return -((-sum + (div >> 1)) / div);
  };

  auto makeCandidate = [&](int laserIdx, int mode, int histIdx) {
    Candidate cand;
    cand.mode = mode;
    cand.histIdx = histIdx;
    cand.valid = true;

    const auto& same = reconState[laserIdx];
    const auto* up = laserIdx > 0 ? &reconState[laserIdx - 1] : nullptr;
    const auto* down = laserIdx + 1 < kNumRings ? &reconState[laserIdx + 1] : nullptr;

    switch (mode) {
    case kPredZero:
      cand.pred = 0;
      cand.predR = 0;
      cand.boundary = false;
      break;

    case kPredSameRingLast:
      cand.valid = same.hasLast();
      if (cand.valid) {
        cand.pred = same.last;
        cand.predR = same.lastR;
        cand.boundary = same.hasPrev()
          && std::abs(same.lastR - same.prevR) > boundaryThreshold();
      }
      break;

    case kPredSameRingLinear2:
      cand.valid = same.hasPrev();
      if (cand.valid) {
        cand.pred = same.last * 2 - same.prev;
        cand.predR = computeRApprox(cand.pred);
        cand.boundary = std::abs(same.lastR - same.prevR) > boundaryThreshold();
      }
      break;

    case kPredCrossRingUp:
      cand.valid = up && up->hasLast();
      if (cand.valid) {
        cand.pred = up->last;
        cand.predR = up->lastR;
        cand.boundary = same.hasLast()
          && std::abs(up->lastR - same.lastR) > boundaryThreshold();
      }
      break;

    case kPredCrossRingDown:
      cand.valid = down && down->hasLast();
      if (cand.valid) {
        cand.pred = down->last;
        cand.predR = down->lastR;
        cand.boundary = same.hasLast()
          && std::abs(down->lastR - same.lastR) > boundaryThreshold();
      }
      break;

    case kPredFusedPrevPhi: {
      int sum[3] = {};
      int count = 0;
      bool boundary = false;
      cand.valid = same.hasLast();
      if (same.hasLast()) {
        for (int k = 0; k < 3; k++)
          sum[k] += same.last[k];
        count++;
      }
      if (up && up->hasLast()) {
        for (int k = 0; k < 3; k++)
          sum[k] += up->last[k];
        count++;
        boundary |= same.hasLast()
          && std::abs(up->lastR - same.lastR) > boundaryThreshold();
      }
      if (down && down->hasLast()) {
        for (int k = 0; k < 3; k++)
          sum[k] += down->last[k];
        count++;
        boundary |= same.hasLast()
          && std::abs(down->lastR - same.lastR) > boundaryThreshold();
      }
      cand.valid = count >= 2;
      if (cand.valid) {
        for (int k = 0; k < 3; k++)
          cand.pred[k] = roundedDiv(sum[k], count);
        cand.predR = computeRApprox(cand.pred);
        cand.boundary = boundary;
      }
      break;
    }

    case kPredSameRingCubic3:
      cand.valid = same.hasPrev2();
      if (cand.valid) {
        cand.pred = same.last * 3 - same.prev * 3 + same.prev2;
        cand.predR = computeRApprox(cand.pred);
        cand.boundary =
          std::abs(same.lastR - same.prevR) > boundaryThreshold()
          || std::abs(same.prevR - same.prev2R) > boundaryThreshold();
      }
      break;

    case kPredCrossRingCorrected:
      cand.valid = same.hasPrev() && ((up && up->hasLast()) || (down && down->hasLast()));
      if (cand.valid) {
        const auto* adj = up && up->hasLast() ? up : down;
        point_t delta = same.last - same.prev;
        const int limit = std::max(boundaryThreshold() >> 1, same.lastR >> 3);
        for (int k = 0; k < 3; k++)
          delta[k] = PCCClip(delta[k], -limit, limit);
        cand.pred = adj->last + delta;
        cand.predR = computeRApprox(cand.pred);
        cand.boundary = std::abs(adj->lastR - same.lastR) > boundaryThreshold();
      }
      break;

    case kPredSameRingHist:
      cand.valid = same.count >= 2 && histIdx < std::min(same.count, kHistTestCount);
      if (cand.valid) {
        cand.pred = same.histPts[histIdx];
        cand.predR = same.histR[histIdx];
        cand.boundary = histIdx + 1 < same.count
          && std::abs(same.histR[histIdx] - same.histR[histIdx + 1]) > boundaryThreshold();
      }
      break;

    default:
      cand.valid = false;
      break;
    }

    cand.rangeClass = computeRangeClass(cand.predR);
    return cand;
  };

  int codedIdx = 0;
  std::array<int64_t, kCartesianPredModes> modeHist = {};
  std::array<double, kCartesianPredModes> modeBitsAccum = {};
  std::array<double, 3> residualBitsAccum = {};
  std::array<int64_t, 3> zeroHist = {};
  std::array<int64_t, kHistTestCount> histIdxHist = {};
  std::array<int64_t, kCartesianPredModes> modeUseBitOnly = {};
  std::array<std::vector<float>, kCartesianPredModes> modeDeltaWin = {};
  std::array<std::vector<float>, kCartesianPredModes> modeDeltaAll = {};
  static bool printedPenaltyTable = false;
  if (!printedPenaltyTable) {
    std::cout << "  Flat predgeom modePenalty:";
    for (int mode = 0; mode < kCartesianPredModes; mode++)
      std::cout << " m" << mode << "=" << modePenalty(mode);
    std::cout << std::endl;
    printedPenaltyTable = true;
  }

  for (int p = 0; p < numPoints; p++) {
    const auto& curr = cloud[p];
    const int laserIdx = clampLaserIdx(cloud, p);
    const int currR = computeRApprox(curr);
    int modeRangeClass = 0;
    bool modeBoundary = false;
    if (reconState[laserIdx].hasLast()) {
      modeRangeClass = computeRangeClass(reconState[laserIdx].lastR);
      modeBoundary = reconState[laserIdx].hasPrev()
        && std::abs(reconState[laserIdx].lastR - reconState[laserIdx].prevR)
             > boundaryThreshold();
    }

    Candidate best;
    point_t bestResidual = 0;
    float bestBits = std::numeric_limits<float>::max();
    std::array<float, kCartesianPredModes * kHistTestCount> analysisBits = {};
    std::array<int, kCartesianPredModes * kHistTestCount> analysisModes = {};
    int analysisCount = 0;
    for (int mode = 0; mode < kCartesianPredModes; mode++) {
      const int histLimit = mode == kPredSameRingHist
        ? std::min(reconState[laserIdx].count, kHistTestCount)
        : 1;

      for (int histIdx = 0; histIdx < histLimit; histIdx++) {
        auto cand = makeCandidate(laserIdx, mode, histIdx);
        if (!cand.valid)
          continue;

        if ((mode == kPredCrossRingUp || mode == kPredCrossRingDown
             || mode == kPredCrossRingCorrected)
            && std::abs(currR - cand.predR) > (boundaryThreshold() << 1)) {
          continue;
        }

        const point_t residual = curr - cand.pred;
        float bitsNoPenalty = enc.estimateModeBits(
          mode, ctxGroupForLaser(laserIdx), modeRangeClass, modeBoundary);
        if (mode == kPredSameRingHist) {
          bitsNoPenalty += enc.estimateHistIdxBits(
            histIdx, ctxGroupForLaser(laserIdx), modeRangeClass, modeBoundary);
        }
        bitsNoPenalty += enc.estimatePredGeomBits(
          residual, ctxGroupForLaser(laserIdx), mode, cand.rangeClass, cand.boundary);

        if (kEnableModePenaltyAnalysis && analysisCount < int(analysisBits.size())) {
          analysisBits[analysisCount] = bitsNoPenalty;
          analysisModes[analysisCount] = mode;
          analysisCount++;
        }

        float bits = bitsNoPenalty + modePenalty(mode);

        if (bits < bestBits) {
          best = cand;
          bestResidual = residual;
          bestBits = bits;
        }
      }
    }

    if (kEnableModePenaltyAnalysis && analysisCount > 0) {
      int bestIdx = 0;
      int secondIdx = -1;
      for (int i = 1; i < analysisCount; i++) {
        if (analysisBits[i] < analysisBits[bestIdx]) {
          secondIdx = bestIdx;
          bestIdx = i;
        } else if (
          secondIdx < 0 || analysisBits[i] < analysisBits[secondIdx]) {
          secondIdx = i;
        }
      }

      const int winMode = analysisModes[bestIdx];
      const float bMin = analysisBits[bestIdx];
      const float b2 = secondIdx >= 0 ? analysisBits[secondIdx] : bMin;
      modeUseBitOnly[winMode]++;
      modeDeltaWin[winMode].push_back(std::max(0.0F, b2 - bMin));
      for (int i = 0; i < analysisCount; i++)
        modeDeltaAll[analysisModes[i]].push_back(std::max(0.0F, analysisBits[i] - bMin));
    }

    const int ctxGroup = ctxGroupForLaser(laserIdx);

    enc.encodeMode(best.mode, ctxGroup, modeRangeClass, modeBoundary);
    if (best.mode == kPredSameRingHist) {
      enc.encodeHistIdx(best.histIdx, ctxGroup, modeRangeClass, modeBoundary);
    }
    enc.encodePredGeom(
      bestResidual, ctxGroup, best.mode, best.rangeClass, best.boundary);

    const point_t reconPoint = best.pred + bestResidual;
    reconState[laserIdx].push(reconPoint, computeRApprox(reconPoint));

    modeHist[best.mode]++;
    if (best.mode == kPredSameRingHist && best.histIdx < kHistTestCount)
      histIdxHist[best.histIdx]++;
    modeBitsAccum[best.mode] += enc.estimateModeBits(
      best.mode, ctxGroup, modeRangeClass, modeBoundary);
    if (best.mode == kPredSameRingHist) {
      modeBitsAccum[best.mode] += enc.estimateHistIdxBits(
        best.histIdx, ctxGroup, modeRangeClass, modeBoundary);
    }
    for (int k = 0; k < 3; k++)
      zeroHist[k] += bestResidual[k] == 0;
    const auto residualBits = enc.estimatePredGeomBits(
      bestResidual, ctxGroup, best.mode, best.rangeClass, best.boundary);
    residualBitsAccum[0] += residualBits;

    codedOrder[codedIdx] = p;
    codedIdx++;
  }
  enc.encodeEndOfTreesFlag(true);

  // save the context state for re-use by a future slice if required
  ctxtMem = enc.getCtx();

  // Resize outCloud to match only the encoded points.
  outCloud.resize(codedIdx);

  // put points in output cloud in coded order (identity in flat mode)
  for (int i = 0; i < codedIdx; i++) {
    auto srcIdx = codedOrder[i];
    outCloud[i] = cloud[srcIdx];
    if (cloud.hasLaserAngles())
      outCloud.setLaserAngle(i, cloud.getLaserAngle(srcIdx));
    if (cloud.hasColors())
      outCloud.setColor(i, cloud.getColor(srcIdx));
    if (cloud.hasReflectances())
      outCloud.setReflectance(i, cloud.getReflectance(srcIdx));
  }

  std::cout << "  Flat predgeom stats:" << std::endl;
  std::cout << "    Mode hist:";
  for (int mode = 0; mode < kCartesianPredModes; mode++)
    std::cout << " m" << mode << "=" << modeHist[mode];
  std::cout << std::endl;
  std::cout << "    Zero-rate:";
  for (int k = 0; k < 3; k++)
    std::cout << " " << char('x' + k) << "="
              << std::fixed << std::setprecision(3)
              << (codedIdx ? double(zeroHist[k]) / codedIdx : 0.0);
  std::cout << std::defaultfloat << std::endl;
  std::cout << "    HistIdx:";
  for (int i = 0; i < kHistTestCount; i++)
    std::cout << " h" << i << "=" << histIdxHist[i];
  std::cout << std::endl;

  if (kEnableModePenaltyAnalysis) {
    constexpr float kAlpha = 0.5F;
    constexpr float kBeta = 0.25F;
    constexpr int kPmax = 48;
    std::array<int, kCartesianPredModes> suggestedPenalty = {};
    for (int mode = 0; mode < kCartesianPredModes; mode++) {
      const float q70Win = quantile(modeDeltaWin[mode], 0.70F);
      const float q50All = quantile(modeDeltaAll[mode], 0.50F);
      int penalty = int(q70Win * kAlpha + q50All * kBeta + 0.5F);
      penalty = std::max(0, std::min(kPmax, penalty));
      suggestedPenalty[mode] = penalty;
    }

    const int maxSameRing = std::max(
      std::max(suggestedPenalty[kPredSameRingLast], suggestedPenalty[kPredSameRingLinear2]),
      std::max(suggestedPenalty[kPredSameRingCubic3], suggestedPenalty[kPredSameRingHist]));
    suggestedPenalty[kPredCrossRingUp] =
      std::max(suggestedPenalty[kPredCrossRingUp], maxSameRing);
    suggestedPenalty[kPredCrossRingDown] =
      std::max(suggestedPenalty[kPredCrossRingDown], maxSameRing);
    suggestedPenalty[kPredCrossRingCorrected] =
      std::max(suggestedPenalty[kPredCrossRingCorrected], maxSameRing);

    int maxOther = 0;
    for (int mode = 1; mode < kCartesianPredModes; mode++)
      maxOther = std::max(maxOther, suggestedPenalty[mode]);
    suggestedPenalty[kPredZero] = std::min(kPmax, std::max(suggestedPenalty[kPredZero], maxOther + 1));

    std::cout << "    Penalty analysis (single-scene):" << std::endl;
    for (int mode = 0; mode < kCartesianPredModes; mode++) {
      std::cout << "      m" << mode << " use=" << modeUseBitOnly[mode]
                << " q70win=" << std::fixed << std::setprecision(2)
                << quantile(modeDeltaWin[mode], 0.70F)
                << " q50all=" << quantile(modeDeltaAll[mode], 0.50F)
                << " -> p=" << suggestedPenalty[mode] << std::endl;
    }
    std::cout << std::defaultfloat;
  }

  swap(cloud, outCloud);
}


// ============================================================================
}  // namespace pcc
