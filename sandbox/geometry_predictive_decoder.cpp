
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
#include "hls.h"
#include "quantization.h"

#include <vector>

namespace pcc {

//============================================================================

// Max ring groups for context array sizing (compile-time)
static const int kMaxRingGroups = 32;
static const int kNumRings = 32;
static const int kHistSize = 3;

class PredGeomDecoder : protected PredGeomContexts {
public:
  PredGeomDecoder(const PredGeomDecoder&) = delete;
  PredGeomDecoder& operator=(const PredGeomDecoder&) = delete;

  PredGeomDecoder(
    const GeometryParameterSet&,
    const GeometryBrickHeader& gbh,
    const PredGeomContexts& ctxtMem,
    EntropyDecoder* aed,
    int numRingGroups = 1);

  /**
   * decodes a sequence of decoded geometry trees.
   * @returns the number of points decoded.
   */
  int decode(
    int numPoints,
    Vec3<int32_t>* outputPoints,
    std::vector<Vec3<int32_t>>* reconSphPos,
    PredGeomPredictor& refFrameSph,
    PredGeomPredictor& refFrameSph2);

  /**
   * decodes a single predictive geometry tree.
   * @returns the number of points decoded.
   */
  int decodeTree(
    Vec3<int32_t>* outA, Vec3<int32_t>* outB, PredGeomPredictor& refFrameSph, PredGeomPredictor& refFrameSph2);

  const PredGeomContexts& getCtx() const { return *this; }

  //==================================================
  Vec3<int32_t> decodePredGeom(const int group);
  int decodeHistIdx(const int group);
  int decodeMode(const int group);
  bool decodeEndOfTreesFlag();
  //==================================================

private:
  int decodeNumDuplicatePoints();
  int decodeNumChildren();
  GPredicter::Mode decodePredMode();
  int decodePredIdx();
  int32_t decodeResPhi(int predIdx, int boundPhi, bool interFlag
    , int refNodeIdx
  );
  int32_t decodeResR(int multiplier, int predIdx, const bool interFlag
    , int refNodeIdx
  );

  Vec3<int32_t> decodeResidual(
    int mode,
    int multiplier,
    int rPred,
    int* azimuthSpeed,
    int predIdx,
    bool interFlag
    , int refNodeIdx
  );

  Vec3<int32_t> decodeResidual2();
  int32_t decodePhiMultiplier(GPredicter::Mode mode, bool interFlag
    , int refNodeIdx
    , int predIdx
  );
  bool decodeInterFlag(const uint8_t interFlagBuffer
  );
  int decodeRefNodeIdx(const bool globalMotionEnabled);
  bool decodeRefDirFlag();
  //bool decodeRefNodeFlag();
  int32_t decodeQpOffset();

private:
  EntropyDecoder* _aed;
  std::vector<int32_t> _stack;
  std::vector<int32_t> _nodeIdxToParentIdx;
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

  Vec3<int> _pgeom_resid_abs_log2_bits;

  int _minVal;

  int _maxPredIdx;
  int _thObj;
  int _thQphi;

  // ---- Per-ring-group context arrays (used by decodePredGeom) ----
  int _numRingGroups;
  AdaptiveBitModel _ctxResGt0_g[kMaxRingGroups][3];
  AdaptiveBitModel _ctxSign_g[kMaxRingGroups][3];
  AdaptiveBitModel _ctxNumBits_g[kMaxRingGroups][8][3][31];
  AdaptiveBitModel _ctxHistIdx_g[kMaxRingGroups][kHistSize];
  AdaptiveBitModel _ctxPredMode_g[kMaxRingGroups][4];
};

//============================================================================

PredGeomDecoder::PredGeomDecoder(
  const GeometryParameterSet& gps,
  const GeometryBrickHeader& gbh,
  const PredGeomContexts& ctxtMem,
  EntropyDecoder* aed,
  int numRingGroups)
  : PredGeomContexts(ctxtMem)
  , _aed(aed)
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
  , _pgeom_resid_abs_log2_bits(gbh.pgeom_resid_abs_log2_bits)
  , _azimuthTwoPiLog2(gps.geom_angular_azimuth_scale_log2_minus11 + 12)
  , _minVal(gbh.pgeom_min_radius)
  , _maxPredIdx(gps.predgeom_max_pred_index)
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

  _stack.reserve(1024);
}

//----------------------------------------------------------------------------

int
PredGeomDecoder::decodeNumDuplicatePoints()
{
  bool num_dup_points_gt0 = _aed->decode(_ctxNumDupPointsGt0);
  if (!num_dup_points_gt0)
    return 0;
  return 1 + _aed->decodeExpGolomb(0, _ctxNumDupPoints);
}

//----------------------------------------------------------------------------

int
PredGeomDecoder::decodeNumChildren()
{
  int val = _aed->decode(_ctxNumChildren[0]);
  if (val == 1) {
    val += _aed->decode(_ctxNumChildren[1]);
    if (val == 2)
      val += _aed->decode(_ctxNumChildren[2]);
  }

  return val ^ 1;
}

//----------------------------------------------------------------------------

GPredicter::Mode
PredGeomDecoder::decodePredMode()
{
  int mode = _aed->decode(_ctxPredMode[0]);
  mode = (mode << 1) + _aed->decode(_ctxPredMode[1 + mode]);
  return GPredicter::Mode(mode);
}

//----------------------------------------------------------------------------

int
PredGeomDecoder::decodePredIdx()
{
  int predIdx = 0;
  while (predIdx < _maxPredIdx && _aed->decode(_ctxPredIdx[predIdx]))
    ++predIdx;
  return predIdx;
}

//----------------------------------------------------------------------------

Vec3<int32_t>
PredGeomDecoder::decodeResidual2()
{
  Vec3<int32_t> residual;
  for (int k = 0; k < 3; ++k) {
    int value = _aed->decode(_ctxResidual2GtN[0][k]);
    if (!value) {
      residual[k] = 0;
      continue;
    }

    value += _aed->decode(_ctxResidual2GtN[1][k]);
    if (value == 1) {
      auto sign = _aed->decode(_ctxSign2[k]);
      residual[k] = sign ? -1 : 1;
      continue;
    }

    value += _aed->decodeExpGolomb(0, _ctxEG2Prefix[k], _ctxEG2Suffix[k]);
    auto sign = _aed->decode(_ctxSign2[k]);
    residual[k] = sign ? -value : value;
  }
  return residual;
}

//----------------------------------------------------------------------------

int32_t
PredGeomDecoder::decodePhiMultiplier(GPredicter::Mode mode, const bool interFlag
  , int refNodeIdx
  , int predIdx
)
{
  if (!_geom_angular_mode_enabled_flag)
    return 0;
  int ctxL = interFlag ? (refNodeIdx > 1 ? 1 : 0)
      : (predIdx ? 1 : 0);
  int interCtxIdx = interFlag ? 1 : 0;

  if (!_aed->decode(_ctxPhiGtN[interCtxIdx][ctxL][0]))
  ///if (!_aed->decode(_ctxPhiGtN[interCtxIdx][0]))
    return 0;

  int value = 1;
  value += _aed->decode(_ctxPhiGtN[interCtxIdx][ctxL][1]);
  //value += _aed->decode(_ctxPhiGtN[interCtxIdx][1]);
  if (value == 1) {
    const auto sign = _aed->decode(_ctxSignPhi[interCtxIdx][ctxL]);
    //const auto sign = _aed->decode(_ctxSignPhi[interCtxIdx]);
    return sign ? -1 : 1;
  }

  auto* ctxs = &_ctxResidualPhi[interCtxIdx][ctxL][0] - 1;
  //auto* ctxs = &_ctxResidualPhi[interCtxIdx][0] - 1;
  value = 1;
  for (int n = 3; n > 0; n--)
    value = (value << 1) | _aed->decode(ctxs[value]);
  value ^= 1 << 3;

  if (value == 7)
    value += _aed->decodeExpGolomb(0, _ctxEGPhi[interCtxIdx][ctxL]);
    //value += _aed->decodeExpGolomb(0, _ctxEGPhi[interCtxIdx]);

  const auto sign = _aed->decode(_ctxSignPhi[interCtxIdx][ctxL]);
  //const auto sign = _aed->decode(_ctxSignPhi[interCtxIdx]);
  return sign ? -(value + 2) : (value + 2);
}

//----------------------------------------------------------------------------
bool
PredGeomDecoder::decodeInterFlag(const uint8_t interFlagBuffer)
{
  uint8_t interFlagCtxIdx =
    interFlagBuffer & PredGeomDecoder::interFlagBufferMask;
  return _aed->decode(_ctxInterFlag[interFlagCtxIdx]) ? true : false;
}

//----------------------------------------------------------------------------

int
PredGeomDecoder::decodeRefNodeIdx(const bool globalMotionEnabled)
{
  int refNodeIdx = 0;
  if (globalMotionEnabled)
    refNodeIdx = _aed->decode(_ctxRefNodeIdx[0]);
  refNodeIdx =
    (refNodeIdx << 1) + _aed->decode(_ctxRefNodeIdx[1 + refNodeIdx]);
  return refNodeIdx;
}

//----------------------------------------------------------------------------

bool 
PredGeomDecoder::decodeRefDirFlag()
{
  return _aed->decode(_ctxRefDirFlag) ? true : false;
}

//----------------------------------------------------------------------------

int32_t
PredGeomDecoder::decodeQpOffset()
{
  int dqp = 0;
  if (!_aed->decode(_ctxQpOffsetAbsGt0))
    return 0;

  dqp = _aed->decodeExpGolomb(0, _ctxQpOffsetAbsEgl) + 1;
  int dqp_sign = _aed->decode(_ctxQpOffsetSign);
  return dqp_sign ? -dqp : dqp;
}

//----------------------------------------------------------------------------

bool
PredGeomDecoder::decodeEndOfTreesFlag()
{
  return _aed->decode(_ctxEndOfTrees);
}

//-------------------------------------------------------------------------

int32_t
PredGeomDecoder::decodeResPhi(
  int predIdx, int boundPhi, const bool interFlag, int refNodeIdx)
{
  int interCtxIdx = interFlag ? 1 : 0;
  int ctxL = interFlag ? (refNodeIdx > 1 ? 1 : 0) : (predIdx ? 1 : 0);
  //int ctxL = predIdx ? 1 : 0;

  if (!_aed->decode(_ctxResPhiGTZero[interCtxIdx][ctxL]))
    return 0;

  int absVal = 1;
  absVal += _aed->decode(_ctxResPhiGTOne[interCtxIdx][ctxL]);
  int interEGkCtxIdx = interFlag ? (refNodeIdx > 1 ? 2 : 1) : 0;
  if (absVal == 2)
    absVal += _aed->decodeExpGolomb(
      1, _ctxResPhiExpGolombPre[interEGkCtxIdx],
      _ctxResPhiExpGolombSuf[interEGkCtxIdx]);

  bool sign =
    _aed->decode(_ctxResPhiSign[ctxL][interCtxIdx ? 4 : _resPhiOldSign]);
  _resPhiOldSign = interFlag ? (refNodeIdx > 1 ? 3 : 2) : (sign ? 1 : 0);
  return sign ? -absVal : absVal;
}
//----------------------------------------------------------------------------
int32_t PredGeomDecoder::decodeResR(const int multiplier, const int predIdx, const bool interFlag
  , int refNodeIdx
)
{
  const int interCtx = interFlag;
  int ctxL = interFlag ? (refNodeIdx > 1 ? 1 : 0)
      : (predIdx ? 1 : 0);
  //int ctxL = predIdx == 0 /* parent */;
  int ctxLR = ctxL + (interFlag ? (abs(multiplier) > 2 ? 2 : 0)
      : (abs(multiplier) > _thQphi ? 2 : 0));
  //int ctxLR = ctxL + (multiplier ? 2 : 0);

  if (!_aed->decode(_ctxResRGTZero[interCtx][ctxLR]))
    return 0;

  int absVal = 1;
  absVal += _aed->decode(_ctxResRGTOne[interCtx][ctxLR]);
  if (absVal == 2)
    absVal += _aed->decode(_ctxResRGTTwo[interCtx][ctxLR]);
  if (absVal == 3)
    absVal += _aed->decodeExpGolomb(
      2, _ctxResRExpGolombPre[interCtx][ctxLR],
      _ctxResRExpGolombSuf[interCtx][ctxLR]);

  int ctxR =
    (_precAzimuthStepDelta ? 4 : 0) + (multiplier ? 2 : 0) + _precSignR;

  bool sign = _aed->decode(
    _ctxResRSign[interCtx ? 2 : _prevInterFlag][ctxL][ctxR]);
  _precSignR = sign;
  _precAzimuthStepDelta = multiplier;
  _prevInterFlag = interFlag;
  return sign ? -absVal : absVal;
}
//----------------------------------------------------------------------------
Vec3<int32_t>
PredGeomDecoder::decodeResidual(int mode, int multiplier, int rPred, int* azimuthSpeed, int predIdx, const bool interFlag
  , int refNodeIdx
)
{
  Vec3<int32_t> residual;
  int interCtxIdx = interFlag ? 1 : 0;

  *azimuthSpeed = _geomAngularAzimuthSpeed;

  int k = 0;

  if (_azimuth_scaling_enabled_flag) {
    // N.B. mode is always 1 with _azimuth_scaling_enabled_flag
    residual[0] = decodeResR(multiplier, predIdx, interFlag
      , refNodeIdx
    );

    int r = rPred + residual[0] << 3;
    auto speedTimesR = int64_t(_geomAngularAzimuthSpeed) * r;
    int phiBound = divExp2RoundHalfInf(speedTimesR, _azimuthTwoPiLog2 + 1);
    residual[1] = decodeResPhi(predIdx, phiBound, interFlag
      , refNodeIdx
    );
    if (r && !phiBound) {
      const int32_t pi = 1 << (_azimuthTwoPiLog2 - 1);
      int32_t speedTimesR32 = speedTimesR;
      while (speedTimesR32 < pi) {
        speedTimesR32 <<= 1;
        *azimuthSpeed <<= 1;
      }
    }
    k = 2;
  }

  for (int ctxIdx = 0; k < 3; ++k) {
    // The last component (delta laseridx) isn't coded if there is one laser
    if (_geom_angular_mode_enabled_flag && _numLasers == 1 && k == 2) {
      residual[k] = 0;
      continue;
    }

    if (!_aed->decode(_ctxResGt0[interCtxIdx][k])) {
      residual[k] = 0;
      continue;
    }

    AdaptiveBitModel* ctxs = &_ctxNumBits[interCtxIdx][ctxIdx][k][0] - 1;
    int32_t numBits = 1;
    for (int n = 0; n < _pgeom_resid_abs_log2_bits[k]; n++)
      numBits = (numBits << 1) | _aed->decode(ctxs[numBits]);
    numBits ^= 1 << _pgeom_resid_abs_log2_bits[k];

    if (!k && !_geom_angular_mode_enabled_flag)
      ctxIdx = std::min(4, (numBits + 1) >> 1);

    int32_t res = 0;
    --numBits;
    if (numBits <= 0) {
      res = 2 + numBits;
    } else {
      res = 1 + (1 << numBits);
      for (int i = 0; i < numBits; ++i) {
        res += _aed->decode() << i;
      }
    }

    int sign = 0;

    if (mode || k) {
      sign = _aed->decode(_ctxSign[interCtxIdx][k]);

    }
    residual[k] = sign ? -res : res;
  }

  return residual;
}

//----------------------------------------------------------------------------

int
PredGeomDecoder::decodeTree(
  Vec3<int32_t>* outA, Vec3<int32_t>* outB, PredGeomPredictor& refFrameSph, PredGeomPredictor& refFrameSph2)
{
  QuantizerGeom quantizer(_sliceQp);
  int nodesUntilQpOffset = 0;
  int nodeCount = 0;
  int prevNodeIdx = -1;
  uint8_t interFlagBuffer = 0;

  _stack.push_back(-1);

  const int MaxNPred = kPTEMaxPredictorIndex + 1;
  const int NPred = _maxPredIdx + 1;

  std::array<std::array<int, 2>, MaxNPred> preds = {};
  const bool frameMovingState =
    refFrameSph.isInterEnabled() && refFrameSph.getFrameMovingState();
  const bool frameMovingState2 =
    refFrameSph2.isInterEnabled() && refFrameSph2.getFrameMovingState();

  while (!_stack.empty()) {
    auto parentNodeIdx = _stack.back();
    _stack.pop_back();
    bool isInterEnabled = refFrameSph.isInterEnabled() && prevNodeIdx >= 0;
    bool isInterEnabled2 = refFrameSph2.isInterEnabled() && prevNodeIdx >= 0;

    if (_geom_scaling_enabled_flag && !nodesUntilQpOffset--) {
      int qpOffset = decodeQpOffset() << _geom_qp_multiplier_log2;
      int qp = _sliceQp + qpOffset;
      quantizer = QuantizerGeom(qp);
      nodesUntilQpOffset = _qpOffsetInterval;
    }

    // allocate point in traversal order (depth first)
    auto curNodeIdx = nodeCount++;
    _nodeIdxToParentIdx[curNodeIdx] = parentNodeIdx;

    int numDuplicatePoints = 0;
    if (!_geom_unique_points_flag)
      numDuplicatePoints = decodeNumDuplicatePoints();
    int numChildren = decodeNumChildren();

    bool interFlag = false, refDirFlag = false;
    int refNodeIdx = 0;
    int numRef = isInterEnabled + isInterEnabled2;
    //bool interFlag = false, refNodeFlag = false;
    if (isInterEnabled)
      interFlag = decodeInterFlag(interFlagBuffer);
    
    if (interFlag && (numRef > 1))
      refDirFlag = decodeRefDirFlag();

    if (interFlag)
      refNodeIdx = decodeRefNodeIdx(refFrameSph.getGlobalMotionEnabled());
      //refNodeFlag = decodeRefNodeFlag();

    auto mode = GPredicter::Mode(1);
    int predIdx = 0;
    if (!interFlag)
      if (_azimuth_scaling_enabled_flag)
        predIdx = decodePredIdx();
      else
        mode = decodePredMode();
    int qphi = decodePhiMultiplier(mode, interFlag
      , refNodeIdx
      , predIdx
    );

    point_t pred;
    if (!interFlag || prevNodeIdx == -1) {
      auto predicter = makePredicter(curNodeIdx, mode, _minVal, [&](int idx) {
        return _nodeIdxToParentIdx[idx];
      });

      pred = predicter.predict(outA, mode, _geom_angular_mode_enabled_flag);

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
      const auto prevPos = outA[prevNodeIdx];
      const auto parentPos = outA[_nodeIdxToParentIdx[curNodeIdx]];
      auto& refFrame = refDirFlag ? refFrameSph2 : refFrameSph;
      auto& frameMoving = refDirFlag ? frameMovingState2 : frameMovingState;
      
      const auto interPred =
        refFrame.getInterPred(prevPos[1], prevPos[2], refNodeIdx);
      assert(interPred.first);
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

    int azimuthSpeed;

    auto residual =
      decodeResidual(mode, qphi, pred[0], &azimuthSpeed, predIdx, interFlag
        , refNodeIdx
      );

    if (!_geom_angular_mode_enabled_flag)
      for (int k = 0; k < 3; k++)
        residual[k] = int32_t(quantizer.scale(residual[k]));

    if (_geom_angular_mode_enabled_flag && !_azimuth_scaling_enabled_flag)
      if (mode >= 0)
        pred[1] += qphi * _geomAngularAzimuthSpeed;

    if (_azimuth_scaling_enabled_flag) {
      auto r = (pred[0] + residual[0]) << 3;
      if (r)
        pred[1] += qphi * azimuthSpeed;
      else
        r = 1;
      int32_t rInvLog2Scale;
      int64_t rInv = recipApprox(r, rInvLog2Scale);
      residual[1] =
        divExp2(residual[1] * rInv, rInvLog2Scale - _azimuthTwoPiLog2);
    }
    auto pos = pred + residual;

    if (_azimuth_scaling_enabled_flag) {
      if (pos[1] < -1 << (_azimuthTwoPiLog2 - 1))
        pos[1] += (1 << _azimuthTwoPiLog2);
      if (pos[1] >= 1 << (_azimuthTwoPiLog2 - 1))
        pos[1] -= (1 << _azimuthTwoPiLog2);
    }

    if (!_geom_angular_mode_enabled_flag)
      for (int k = 0; k < 3; k++)
        pos[k] = std::max(0, pos[k]);
    outA[curNodeIdx] = pos;

    if (_azimuth_scaling_enabled_flag) {
      bool flagNewObject =
        (interFlag ? std::abs(pos[0] - preds[0][0]) : std::abs(residual[0]))
        > _thObj;
      int predBIdx = flagNewObject ? NPred - 1 : predIdx;
      for (int i = predBIdx; i > 0; i--)
        preds[i] = preds[i - 1];
      preds[0][0] = pos[0];
      preds[0][1] = pos[1];
    }

    // convert pos from spherical to cartesian, add secondary residual
    if (_geom_angular_mode_enabled_flag) {
      if (!_predgeometry_residual2_disabling_enabled_flag) {
        residual = decodeResidual2();
      } else {
        residual = 0;
      }

      for (int k = 0; k < 3; k++)
        residual[k] = int32_t(quantizer.scale(residual[k]));

      assert(pos[2] < _numLasers && pos[2] >= 0);
      pred = origin + _sphToCartesian(pos);
      outB[curNodeIdx] = pred + residual;
      for (int k = 0; k < 3; k++)
        outB[curNodeIdx][k] = std::max(0, outB[curNodeIdx][k]);
    }

    // copy duplicate point output
    for (int i = 0; i < numDuplicatePoints; i++, nodeCount++) {
      outA[nodeCount] = outA[curNodeIdx];
      outB[nodeCount] = outB[curNodeIdx];
    }

    for (int i = 0; i < numChildren; i++)
      _stack.push_back(curNodeIdx);

    prevNodeIdx = curNodeIdx;
    interFlagBuffer = (interFlagBuffer << 1) | (interFlag ? 1 : 0);
  }

  return nodeCount;
}

//----------------------------------------------------------------------------

int
PredGeomDecoder::decode(
  int numPoints,
  Vec3<int32_t>* outputPoints,
  std::vector<Vec3<int32_t>>* reconPosSph, PredGeomPredictor& refFrameSph,
  PredGeomPredictor& refFrameSph2)
{
  if (_azimuth_scaling_enabled_flag && _maxPredIdx > kPTEMaxPredictorIndex)
    std::runtime_error("gps.predgeom_max_pred_index is out of bound");

  _nodeIdxToParentIdx.resize(numPoints);

  // An intermediate buffer used for reconstruction of the spherical
  // co-ordinates.
  auto* reconA = outputPoints;
  std::vector<Vec3<int32_t>> sphericalPos;
  if (_geom_angular_mode_enabled_flag) {
    if (reconPosSph)
      std::swap(*reconPosSph, sphericalPos);
    sphericalPos.resize(numPoints);
    reconA = sphericalPos.data();
  }

  int32_t pointCount = 0;
  do {
    auto numSubtreePoints = decodeTree(reconA, outputPoints, refFrameSph, refFrameSph2);
    outputPoints += numSubtreePoints;
    reconA += numSubtreePoints;
    pointCount += numSubtreePoints;
  } while (!decodeEndOfTreesFlag());

  if (reconPosSph)
    std::swap(*reconPosSph, sphericalPos);

  return pointCount;
}

// ============================================================================
// Predictive Geometry Coding Cartesian
// ============================================================================

int PredGeomDecoder::decodeHistIdx(const int group) {
  const int g = group;

  int histIdx = 0;
  while (histIdx < kHistSize && _aed->decode(_ctxHistIdx_g[g][histIdx])) {
    ++histIdx;
  }
  return histIdx;
}

int PredGeomDecoder::decodeMode(const int group) {
  const int g = group;

  int mode = 0;
  while (mode < 3 && _aed->decode(_ctxPredMode_g[g][mode])) {
    ++mode;
  }
  return mode;
}

Vec3<int32_t>
PredGeomDecoder::decodePredGeom(const int group)
{
  const int g = group;

  Vec3<int32_t> residual;
  int k = 0;
  for (int ctxIdx = 0; k < 3; k++) {
    if (!_aed->decode(_ctxResGt0_g[g][k])) {
      residual[k] = 0;
      continue;
    }

    int32_t numBits = 0;
    AdaptiveBitModel* ctxs = &_ctxNumBits_g[g][ctxIdx][k][0] - 1;
    int ctx = 1;
    for (int n = _pgeom_resid_abs_log2_bits[k] - 1; n >= 0; n--) {
      auto bin = _aed->decode(ctxs[ctx]);
      numBits = (numBits << 1) | bin;
      ctx = (ctx << 1) | bin;
    }

    ctxIdx = std::min(7, (numBits + 1) >> 1);

    int32_t value = 0;
    int32_t numBitsExtra = numBits - 1;
    for (int32_t i = 0; i < numBitsExtra; ++i) {
      value |= _aed->decode() << i;
    }
    if (numBits > 0) {
      value += (1 << numBitsExtra);
    }
    
    int32_t res = value + 1;
    bool sign = _aed->decode(_ctxSign_g[g][k]);
    residual[k] = sign ? -res : res;
  }
  return residual;
}

//============================================================================

void
decodePredictiveGeometry(
  const GeometryParameterSet& gps,
  const GeometryBrickHeader& gbh,
  PCCPointSet3& pointCloud, std::vector<Vec3<int32_t>>* reconPosSph,
  PredGeomPredictor& refFrameSph,
  PredGeomPredictor& refFrameSph2,
  PredGeomContexts& ctxtMem,
  EntropyDecoder& aed)
{
  PredGeomDecoder dec(gps, gbh, ctxtMem, &aed);
  refFrameSph.init(
    gps.interAzimScaleLog2, gps.numLasers(), gps.globalMotionEnabled,
    gps.resamplingEnabled);
  refFrameSph2.init(
    gps.interAzimScaleLog2, gps.numLasers(), gps.globalMotionEnabled,
    gps.resamplingEnabled);
  dec.decode(
    gbh.footer.geom_num_points_minus1 + 1, &pointCloud[0], reconPosSph,
    refFrameSph, refFrameSph2);
  ctxtMem = dec.getCtx();
}

//============================================================================

void decodePredictiveGeometry(
  const GeometryParameterSet& gps,
  const GeometryBrickHeader& gbh,
  PCCPointSet3& cloud,
  PredGeomContexts& ctxtMem,
  EntropyDecoder* arithmeticDecoder,
  int numGroups)
{
  auto numPoints = gbh.footer.geom_num_points_minus1 + 1;

  cloud.addRemoveAttributes(cloud.hasColors(), cloud.hasReflectances());
  cloud.resize(numPoints);

  PredGeomDecoder dec(gps, gbh, ctxtMem, arithmeticDecoder, numGroups);

  struct RingHist {
    std::array<point_t, kHistSize> samples = {};
    std::array<int, kHistSize> rApproxs = {};
    int count = 0;
    bool valid = false;

    void push(const point_t& pt, int r) {
      samples[count % kHistSize] = pt;
      rApproxs[count % kHistSize] = r;
      count++;
      valid = true;
    }

    point_t getNormPred(int idx) const {
      return samples[idx];
    }

    point_t getModePred(int mode) const {
      point_t p0 = samples[(count - 1 + kHistSize) % kHistSize];
      point_t p1 = samples[(count - 2 + kHistSize) % kHistSize];
      point_t p2 = samples[(count - 3 + kHistSize) % kHistSize];

      if (mode == 0) return p0;
      if (mode == 1) return 2 * p0 - p1;
      return 3 * p0 - 3 * p1 + p2;
    }

    int latestR() const {
      return rApproxs[(count - 1) % kHistSize];
    }
  };

  std::array<RingHist, kNumRings> reconHist = {};

  for (int p = 0; p < numPoints; p++) {
    const int laserIdx = p % kNumRings;

    const int ringsPerGroup = kNumRings / numGroups;
    int ctxGroup = laserIdx / ringsPerGroup;
    if (ctxGroup < 0)
      ctxGroup = 0;
    else if (ctxGroup >= numGroups)
      ctxGroup = numGroups - 1;

    int mode = dec.decodeMode(ctxGroup);
    int histIdx = 0;
    if (mode == 0) {
      histIdx = dec.decodeHistIdx(ctxGroup);
    }

    Vec3<int32_t> residual = dec.decodePredGeom(ctxGroup);

    point_t reconPred = 0;
    if (reconHist[laserIdx].valid) {
      if (mode == 0) {
        reconPred = reconHist[laserIdx].getNormPred(histIdx);
      } else {
        reconPred = reconHist[laserIdx].getModePred(mode - 1);
      }
    }

    point_t reconPoint = reconPred + residual;

    auto rAbsX = std::abs(reconPoint[0]);
    auto rAbsY = std::abs(reconPoint[1]);
    int reconR = std::max(rAbsX, rAbsY) + static_cast<int>(0.414 * std::min(rAbsX, rAbsY));

    reconHist[laserIdx].push(reconPoint, reconR);

    // Keep point order flat without mapping/reordering inside decoder.
    cloud[p] = reconPoint;
  }

  // Consume tree terminator flags until the final true is encountered.
  while (!dec.decodeEndOfTreesFlag()) {
  }

  ctxtMem = dec.getCtx();
}

//============================================================================

}  // namespace pcc
