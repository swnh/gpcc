#pragma once

#include <algorithm>
#include <array>
#include <cstdint>
#include <cstdlib>
#include <stdexcept>

#include "PCCMath.h"
#include "PCCPointSet.h"
#include "geometry.h"

namespace pcc {
namespace flat_predgeom {

static const int kNumRings = 32;
static const int kHistSize = 4;
static const int kCartesianRangeClasses = 3;
static const int kCartesianBoundaryClasses = 2;
static const int kCartesianPredModes = 4;
static const int kCartesianPredFamilies = 2;
// Two mode bins per (group, range, boundary):
//   0: mode MSB
//   1: mode LSB conditioned on modeMSB=0
//   2: mode LSB conditioned on modeMSB=1
static const int kCartesianPredModeTreeNodes = 3;

// Mode mapping table for fixed 2-bit mode header:
//   mode bits | semantic
//      00     | Zero
//      01     | SameRingLast
//      10     | CrossRingUp
//      11     | CrossRingDown
enum CartesianPredMode
{
  kPredZero = 0,
  kPredSameRingLast = 1,
  kPredCrossRingUp = 2,
  kPredCrossRingDown = 3,
};

struct ModeBits {
  int familyBit;
  int groupBits;
};

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
  point_t pred = 0;
  int predR = 0;
  int rangeClass = 0;
  bool boundary = false;
  bool valid = true;
};

struct ModeContextKey {
  int rangeClass = 0;
  bool boundary = false;
};

struct ResidualContextKey {
  int rangeClass = 0;
  bool boundary = false;
  int modeFamily = 0;
};

inline int
boundaryThreshold()
{
  return 1024;
}

inline int
clampLaserIdx(const PCCPointSet3& cloud, int idx)
{
  if (!cloud.hasLaserAngles())
    return std::max(0, std::min(kNumRings - 1, idx));
  return std::max(0, std::min(kNumRings - 1, cloud.getLaserAngle(idx)));
}

inline uint32_t computeRApprox(const point_t& pt)
{
    const uint32_t ax = (pt[0] < 0) ? uint32_t(-pt[0]) : uint32_t(pt[0]);
    const uint32_t ay = (pt[1] < 0) ? uint32_t(-pt[1]) : uint32_t(pt[1]);

    const uint32_t maxv = (ax > ay) ? ax : ay;
    const uint32_t minv = (ax > ay) ? ay : ax;

    return maxv + (((minv << 5) + (minv << 4) + (minv << 2) + minv) >> 7);
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
predFamily(int mode)
{
  return (mode >> 1) & 1;
}

inline ModeBits
modeToBits(int mode)
{
  if (mode < 0 || mode >= kCartesianPredModes)
    throw std::runtime_error("flat predgeom: invalid mode id");
  return {(mode >> 1) & 1, mode & 0b11};
}

inline int
bitsToMode(int modeMsb, int modeLsb)
{
  return ((modeMsb & 1) << 1) | (modeLsb & 1);
}

inline int
modeTreeNodeMsb()
{
  return 0;
}

inline int
modeTreeNodeLsb(int modeMsb)
{
  return 1 + (modeMsb ? 1 : 0);
}

inline int
ctxGroupForLaser(int numGroups, int laserIdx)
{ 
  const int ringsPerGroup = std::max(1, kNumRings / numGroups);
  return std::min(numGroups - 1, std::max(0, laserIdx / ringsPerGroup));
}

inline ModeContextKey
deriveModeContext(const RingState& state)
{
  ModeContextKey key;
  if (!state.hasLast())
    return key;

  key.rangeClass = computeRangeClass(state.lastR);
  key.boundary = state.hasPrev()
    && std::abs(state.lastR - state.prevR) > boundaryThreshold();
  return key;
}

inline ResidualContextKey
deriveResidualContext(const Candidate& cand)
{
  ResidualContextKey key;
  key.rangeClass = cand.rangeClass;
  key.boundary = cand.boundary;
  key.modeFamily = predFamily(cand.mode);
  return key;
}

inline Candidate
makeCandidate(const std::array<RingState, kNumRings>& reconState, int laserIdx, int mode)
{
  Candidate cand;
  cand.mode = mode;

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

  default:
    cand.valid = false;
    break;
  }

  cand.rangeClass = computeRangeClass(cand.predR);
  return cand;
}

inline int64_t
residualL1(const point_t& residual)
{
  return int64_t(std::abs(residual[0])) + int64_t(std::abs(residual[1]))
    + int64_t(std::abs(residual[2]));
}

// Angular-grid predictor state.  These helpers are shared by the sandbox
// encoder and decoder so predictor-relative syntax stays bit-exact.
//
// Grid mapping is *pred-relative*: the integer "column multiplier" dColumn is
// computed as round((curr.phi - pred.phi) / azimuthSpeed) against the
// selected pred's actual phi, not against a global scalePhi frame.  The FIFO
// stores raw phi only; column is never materialized in state.
//
// Pred list is a TMC3-style sticky FIFO of up to kAngularMaxPredList slots.
// On each point, the encoder picks bestIdx and the FIFO update shifts only
// slots [0..bestIdx] — deeper slots persist as long-term exemplars.  When
// |curr.sphR - preds[0].sphR| exceeds a "new-object" threshold, the update
// forces a full-list shift (bestIdx = listSize-1) so the jumped radius lands
// in the tail rather than poisoning the common slots.  Cross-packet state is
// carried seamlessly — nothing resets at packet boundaries.
static const int kAngularMaxPredList = 7;

struct AngularGridPoint {
  AngularGridPoint() = default;
  AngularGridPoint(int32_t r, int32_t ph, int32_t sR, bool v)
    : ring(r), phi(ph), sphR(sR), valid(v)
  {}

  int32_t ring  = 0;
  int32_t phi   = 0;   // raw phi in [0, scalePhi)
  int32_t sphR  = 0;
  bool    valid = false;
};

struct AngularGridState {
  // Sticky FIFO: preds[0] = most recent; slots beyond bestIdx persist across
  // updates until evicted by a new-object shift or by accumulated wins.
  std::array<AngularGridPoint, kAngularMaxPredList> preds = {};
  int count = 0;

  // Sticky update with optional new-object reset.
  void updateSticky(
    const AngularGridPoint& pt, int bestIdx, int listSize, bool newObject)
  {
    const int shiftEnd = newObject ? (listSize - 1) : bestIdx;
    for (int i = shiftEnd; i > 0; --i)
      preds[i] = preds[i - 1];
    preds[0] = pt;
    count = std::min(count + 1, listSize);
  }
};

struct AngularMetrics {
  std::array<int, kAngularMaxPredList> predIdxHist = {};
  int sameRingHit = 0;       // # points where preds[bestIdx].ring == curr.ring
  int ringTransitions = 0;   // # points where curr.ring != prev.ring
  int newObjectHits = 0;     // # points that triggered the new-object reset
};

inline int32_t
positiveMod(int64_t v, int32_t period)
{
  v %= period;
  return int32_t(v < 0 ? v + period : v);
}

inline int32_t
wrapColumn(int64_t column, int32_t maxPtsPerRot)
{
  return positiveMod(column, maxPtsPerRot);
}

inline int32_t
signedModDelta(int64_t curr, int64_t pred, int32_t period)
{
  int32_t d = positiveMod(curr - pred, period);
  if (d > period / 2)
    d -= period;
  return d;
}

inline int32_t
deriveAngularMaxPtsPerRot(int32_t scalePhi, int32_t azimuthSpeed)
{
  return std::max<int32_t>(
    1, int32_t((int64_t(scalePhi) + (azimuthSpeed >> 1)) / azimuthSpeed));
}

inline int32_t
roundDivPow2Signed(int32_t v, int log2Step)
{
  if (log2Step <= 0)
    return v;

  const int64_t half = int64_t(1) << (log2Step - 1);
  const int64_t mag = std::abs(int64_t(v));
  const int64_t q = (mag + half) >> log2Step;
  return v < 0 ? -int32_t(q) : int32_t(q);
}

// Round v / d to nearest integer, signed. d > 0, d may not be power-of-2.
inline int32_t
roundDivSigned(int64_t v, int32_t d)
{
  const int64_t half = d / 2;
  return v >= 0 ? int32_t((v + half) / d) : -int32_t((-v + half) / d);
}

// Decompose signed phi delta into (integer dColumn multiplier of azimuthSpeed,
// sub-step dAzShift residual). Reconstruction: dPhi = dColumn*azSpeed + dAzShift.
struct AngularPhiDelta {
  int32_t dColumn  = 0;
  int32_t dAzShift = 0;
};

inline AngularPhiDelta
computePredRelativePhiDelta(
  int32_t currPhi, int32_t predPhi, int32_t scalePhi, int32_t azimuthSpeed)
{
  AngularPhiDelta out;
  const int32_t dPhi = signedModDelta(currPhi, predPhi, scalePhi);
  out.dColumn  = roundDivSigned(int64_t(dPhi), azimuthSpeed);
  out.dAzShift = int32_t(int64_t(dPhi) - int64_t(out.dColumn) * azimuthSpeed);
  return out;
}

// Rebuild curr phi from a selected pred plus (dColumn, dAzShift).
inline int32_t
reconstructAngularPhi(
  int32_t predPhi, int32_t dColumn, int32_t dAzShift,
  int32_t scalePhi, int32_t azimuthSpeed)
{
  const int64_t raw = int64_t(predPhi)
    + int64_t(dColumn) * azimuthSpeed + int64_t(dAzShift);
  return positiveMod(raw, scalePhi);
}

// Build a raw-FIFO candidate view of length listSize.  Slots beyond
// state.count are zero-bootstrap (valid=true so the RDO still considers
// them).  No ring filtering, no linear extrapolation — the sticky FIFO
// update on the encoder/decoder side does all the specialization.
inline std::array<AngularGridPoint, kAngularMaxPredList>
viewAngularCandidates(const AngularGridState& state, int listSize, int32_t scalePhi)
{
  std::array<AngularGridPoint, kAngularMaxPredList> cands = {};
  const AngularGridPoint bootstrap{0, scalePhi >> 1, 0, true};
  for (int i = 0; i < kAngularMaxPredList; ++i) {
    cands[i] = i < state.count ? state.preds[i] : bootstrap;
  }
  (void)listSize;
  return cands;
}

inline int32_t
angularPhiToArcResidual(int32_t dPhi, int32_t rScaled, int32_t phiLog2)
{
  return int32_t(divExp2RoundHalfInf(int64_t(dPhi) * rScaled, phiLog2));
}

inline int32_t
angularArcToPhiResidual(int32_t arc, int32_t rScaled, int32_t phiLog2)
{
  if (rScaled <= 0)
    return 0;

  int32_t rInvScaleLog2;
  int64_t rInv = recipApprox(rScaled, rInvScaleLog2);
  return int32_t(divExp2(int64_t(arc) * rInv, rInvScaleLog2 - phiLog2));
}

inline int32_t
angularPhiBound(int32_t azimuthSpeed, int32_t rScaled, int32_t phiLog2)
{
  if (rScaled <= 0)
    return 0;
  return int32_t(
    divExp2RoundHalfInf(int64_t(azimuthSpeed) * rScaled, phiLog2 + 1));
}

}  // namespace flat_predgeom
}  // namespace pcc
