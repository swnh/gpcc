/* ============================================================================
 * TMC3 PredGeom Only
 *
 * Simplified TMC3 flow:
 *   input PLY -> encode -> write .bin/.ply -> read .bin -> decode -> write decoded .ply -> verify
 *
 * Features kept:
 *   - Internal processing time (wall + user)
 *   - Reconstructed point cloud output
 *   - Payload buffer / bitstream size reporting
 * ============================================================================ */

#include "TMC3.h"

#include <algorithm>
#include <cassert>
#include <chrono>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <memory>
#include <sstream>
#include <string>
#include <vector>

#include "PayloadBuffer.h"
#include "entropy.h"
#include "flat_predgeom_common.h"
#include "geometry.h"
#include "geometry_predictive.h"
#include "geometry_params.h"
#include "hls.h"
#include "io_hls.h"
#include "io_tlv.h"
#include "pcc_chrono.h"
#include "ply.h"
#include "PCCPointSet.h"
#include "version.h"

using namespace std;
using namespace pcc;

// ============================================================================
// Minimal parameters for the simplified flow
// ============================================================================
struct SimpleParams {
  string inputPlyPath = "/home/swnh/pgc/datasets/nuscenes/v1.0-mini/ply/bin/scene-0061/scene-0061_00.ply";
  string outputBitstreamPath = "/home/swnh/gpcc/experiments/predgeom.bin";
  string reconstructedPath = "/home/swnh/gpcc/experiments/recon.ply";
  string decodedPath = "/home/swnh/gpcc/experiments/decoded.ply";
  string dumpCsvPath;
  string summaryCsvPath = "scripts/flat_qres_qp_summary.csv";
  double inputScale = 1.0;
  int    numGroups  = 1;
  int    flatQp = -1;
  string ratePreset;            // set by --rate (required in angular mode)
  int    arcLog2 = 0;           // set by --rate (azimuth arc quantization shift)
  int    azimuthScaleLog2 = 7;  // set by --rate
  int    azimuthSpeed = 483;    // set by --rate
  int    radiusInvScaleLog2 = 8;// set by --rate
  int    qphiThreshold = 3;     // set by --rate
  bool   rangeCtxEnabled = true;
  bool   boundaryCtxEnabled = true;
  bool   angularMode = false;   // --mode angular (default flat)
  bool   angularLossless = false;
  bool   batch = false;         // --batch (laser-major, HDL-32E: 32 lasers x 12 firings)
  bool   metrics = false;
};

static constexpr int kBatchPointsPerLaser = 12;
static constexpr int kBatchPacketSize = 32 * kBatchPointsPerLaser;
static constexpr bool kAngularAzimuthScaling = true;
static constexpr bool kAngularResidual2Disabled = true;
static constexpr bool kAngularQphiThresholdPresent = true;

// ============================================================================
// Forward declaration — the simplified encodePredictiveGeometry in
// geometry_predictive_encoder.cpp (ring-based, no tree, no inter)
// ============================================================================
namespace pcc {
void encodePredictiveGeometryAngular(
  const PredGeomEncOpts&,
  const GeometryParameterSet&,
  GeometryBrickHeader&,
  PCCPointSet3&,
  PredGeomContexts&,
  EntropyEncoder*,
  int arcQuantLog2,
  bool lossless,
  flat_predgeom::AngularMetrics* metrics = nullptr);

void decodePredictiveGeometryAngular(
  const GeometryParameterSet&,
  const GeometryBrickHeader&,
  PCCPointSet3&,
  PredGeomContexts&,
  EntropyDecoder*,
  int arcQuantLog2,
  bool lossless);

void encodePredictiveGeometry(
  const PredGeomEncOpts& opt,
  const GeometryParameterSet& gps,
  GeometryBrickHeader& gbh,
  PCCPointSet3& cloud,
  PredGeomContexts& ctxtMem,
  EntropyEncoder* arithmeticEncoder,
  int numGroups,
  int flatFixedQp,
  bool rangeCtxEnabled,
  bool boundaryCtxEnabled,
  const std::string& dumpCsvPath);

void decodePredictiveGeometry(
  const GeometryParameterSet& gps,
  const GeometryBrickHeader& gbh,
  PCCPointSet3& cloud,
  PredGeomContexts& ctxtMem,
  EntropyDecoder* arithmeticDecoder,
  int numGroups,
  bool rangeCtxEnabled,
  bool boundaryCtxEnabled);
}

// ============================================================================
static void
printUsage(const char* argv0)
{
  cerr << "Usage: " << argv0
       << " --input <ply> [--mode <flat|angular>] [options]\n"
       << "  Common:  --output <bin> --recon <ply> --decoded <ply>\n"
       << "           --scale N --summary-csv <csv>\n"
       << "  Flat:    --groups <1|2|4|8|16|32> [--flat-qp N] [--range 0|1]\n"
       << "           [--boundary 0|1] [--dump-csv <csv>]\n"
       << "  Angular: --rate <r06|r05|r04|r03|r02|r01>\n"
       << "           [--angular-lossless] [--batch] [--metrics]\n";
}

static bool
parseIntOption(const string& name, const string& value, int* out)
{
  try {
    size_t pos = 0;
    const int parsed = stoi(value, &pos);
    if (pos != value.size())
      throw std::invalid_argument("junk");
    *out = parsed;
    return true;
  } catch (...) {
    cerr << "Error: " << name << " expects an integer, got '" << value << "'\n";
    return false;
  }
}

static bool
parseDoubleOption(const string& name, const string& value, double* out)
{
  try {
    size_t pos = 0;
    const double parsed = stod(value, &pos);
    if (pos != value.size())
      throw std::invalid_argument("junk");
    *out = parsed;
    return true;
  } catch (...) {
    cerr << "Error: " << name << " expects a number, got '" << value << "'\n";
    return false;
  }
}

static bool
parseBool01Option(const string& name, const string& value, bool* out)
{
  if (value == "0") {
    *out = false;
    return true;
  }
  if (value == "1") {
    *out = true;
    return true;
  }
  cerr << "Error: " << name << " expects 0 or 1, got '" << value << "'\n";
  return false;
}

// ============================================================================
static bool
parseSimpleArgs(int argc, char* argv[], SimpleParams& p)
{
  if (argc == 1) {
    printUsage(argv[0]);
    return false;
  }

  bool sawInput = false;
  bool sawRate = false;
  bool usedFlatOnlyFlag = false;
  bool usedAngularOnlyFlag = false;

  auto isPrefix = [](const string& arg, const string& prefix) {
    return arg.rfind(prefix, 0) == 0;
  };

  auto rejectLegacyOption = [&](const string& arg) {
    if (arg == "-i" || isPrefix(arg, "-i=")) {
      cerr << "Error: -i was removed. Use --input <ply>\n";
      return true;
    }
    if (arg == "-o" || isPrefix(arg, "-o=")) {
      cerr << "Error: -o was removed. Use --output <bin>\n";
      return true;
    }
    if (arg == "-r" || isPrefix(arg, "-r=")) {
      cerr << "Error: -r was removed. Use --recon <ply>\n";
      return true;
    }
    if (arg == "-d" || isPrefix(arg, "-d=")) {
      cerr << "Error: -d was removed. Use --decoded <ply>\n";
      return true;
    }
    if (arg == "-s" || isPrefix(arg, "-s=")) {
      cerr << "Error: -s was removed. Use --scale N\n";
      return true;
    }
    if (arg == "-g" || isPrefix(arg, "-g=")) {
      cerr << "Error: -g was removed. Use --groups N\n";
      return true;
    }
    if (arg == "--qp" || isPrefix(arg, "--qp=")) {
      cerr << "Error: --qp was removed. Use --flat-qp N\n";
      return true;
    }
    if (arg == "--angular" || isPrefix(arg, "--angular=")) {
      cerr << "Error: --angular was removed. Use --mode angular\n";
      return true;
    }
    if (arg == "--angular-rate" || isPrefix(arg, "--angular-rate=")) {
      cerr << "Error: --angular-rate was removed. Use --rate <preset>\n";
      return true;
    }
    return false;
  };

  auto applyRatePreset = [&](const string& rate) {
    if (rate == "r06") {
      p.azimuthScaleLog2 = 12; p.azimuthSpeed = 15463;
      p.radiusInvScaleLog2 = 1; p.qphiThreshold = 0;
      p.arcLog2 = 0;
    } else if (rate == "r05") {
      p.azimuthScaleLog2 = 11; p.azimuthSpeed = 7732;
      p.radiusInvScaleLog2 = 2; p.qphiThreshold = 0;
      p.arcLog2 = 0;
    } else if (rate == "r04") {
      p.azimuthScaleLog2 = 9; p.azimuthSpeed = 1934;
      p.radiusInvScaleLog2 = 4; p.qphiThreshold = 0;
      p.arcLog2 = 0;
    } else if (rate == "r03") {
      p.azimuthScaleLog2 = 8; p.azimuthSpeed = 967;
      p.radiusInvScaleLog2 = 5; p.qphiThreshold = 1;
      p.arcLog2 = 0;
    } else if (rate == "r02") {
      p.azimuthScaleLog2 = 7; p.azimuthSpeed = 483;
      p.radiusInvScaleLog2 = 7; p.qphiThreshold = 3;
      p.arcLog2 = 0;
    } else if (rate == "r01") {
      p.azimuthScaleLog2 = 7; p.azimuthSpeed = 483;
      p.radiusInvScaleLog2 = 8; p.qphiThreshold = 3;
      p.arcLog2 = 0;
    } else {
      return false;
    }
    p.ratePreset = rate;
    return true;
  };

  for (int i = 1; i < argc; i++) {
    string arg = argv[i];

    if (arg == "--help") {
      printUsage(argv[0]);
      return false;
    }
    if (rejectLegacyOption(arg))
      return false;

    auto getValue = [&](const string& longName, string* value) {
      const string prefix = longName + "=";
      if (arg.rfind(prefix, 0) == 0) {
        *value = arg.substr(prefix.size());
        return true;
      }
      if (arg == longName && i + 1 < argc) {
        *value = argv[++i];
        return true;
      }
      return false;
    };
    string value;
    if (getValue("--input", &value)) {
      p.inputPlyPath = value;
      sawInput = true;
    }
    else if (getValue("--mode", &value)) {
      if (value == "flat")
        p.angularMode = false;
      else if (value == "angular")
        p.angularMode = true;
      else {
        cerr << "Error: --mode must be flat or angular\n";
        return false;
      }
    }
    else if (getValue("--output", &value))
      p.outputBitstreamPath = value;
    else if (getValue("--recon", &value))
      p.reconstructedPath = value;
    else if (getValue("--decoded", &value))
      p.decodedPath = value;
    else if (getValue("--scale", &value)) {
      if (!parseDoubleOption("--scale", value, &p.inputScale))
        return false;
    }
    else if (getValue("--groups", &value)) {
      if (!parseIntOption("--groups", value, &p.numGroups))
        return false;
      usedFlatOnlyFlag = true;
    }
    else if (getValue("--dump-csv", &value)) {
      p.dumpCsvPath = value;
      usedFlatOnlyFlag = true;
    }
    else if (getValue("--summary-csv", &value))
      p.summaryCsvPath = value;
    else if (getValue("--flat-qp", &value)) {
      if (!parseIntOption("--flat-qp", value, &p.flatQp))
        return false;
      usedFlatOnlyFlag = true;
    }
    else if (getValue("--dr-log2", &value) || getValue("--arc-log2", &value)) {
      cerr << "Error: --dr-log2/--arc-log2 were removed. "
           << "Quantization is controlled by --rate.\n";
      return false;
    }
    else if (getValue("--rate", &value)) {
      if (!applyRatePreset(value)) {
        cerr << "Error: --rate must be one of r06,r05,r04,r03,r02,r01\n";
        return false;
      }
      sawRate = true;
      usedAngularOnlyFlag = true;
    }
    else if (getValue("--range", &value)) {
      if (!parseBool01Option("--range", value, &p.rangeCtxEnabled))
        return false;
      usedFlatOnlyFlag = true;
    }
    else if (getValue("--boundary", &value)) {
      if (!parseBool01Option("--boundary", value, &p.boundaryCtxEnabled))
        return false;
      usedFlatOnlyFlag = true;
    }
    else if (arg == "--angular-lossless") {
      p.angularLossless = true;
      usedAngularOnlyFlag = true;
    }
    else if (arg == "--batch") {
      p.batch = true;
      usedAngularOnlyFlag = true;
    }
    else if (arg == "--metrics") {
      p.metrics = true;
      usedAngularOnlyFlag = true;
    }
    else {
      cerr << "Unknown option: " << arg << "\n";
      return false;
    }
  }

  if (!sawInput || p.inputPlyPath.empty()) {
    cerr << "Error: --input is required\n";
    return false;
  }
  if (p.flatQp < -1) {
    cerr << "Error: --flat-qp must be >= -1\n";
    return false;
  }
  if (p.angularMode) {
    if (!sawRate) {
      cerr << "Error: --mode angular requires --rate <r06|r05|r04|r03|r02|r01>\n";
      return false;
    }
    if (usedFlatOnlyFlag) {
      cerr << "Error: flat-only options (--groups/--flat-qp/--range/--boundary/--dump-csv)"
           << " cannot be used with --mode angular\n";
      return false;
    }
  } else if (sawRate || usedAngularOnlyFlag) {
    cerr << "Error: angular options (--rate/--angular-lossless/--batch/--metrics)"
         << " require --mode angular\n";
    return false;
  }
  return true;
}

// ============================================================================
//  Minimal defaults for the parameter sets used by PredGeomEncoder
// ============================================================================
static void
setupMinimalSPS(SequenceParameterSet& sps)
{
  sps.sps_seq_parameter_set_id = 0;
  sps.frame_ctr_bits = 16;
  sps.slice_tag_bits = 0;
  sps.geometry_axis_order = AxisOrder::kXYZ;
  sps.cabac_bypass_stream_enabled_flag = true;
  sps.bypass_bin_coding_without_prob_update = false;
  sps.entropy_continuation_enabled_flag = false;
  sps.inter_frame_prediction_enabled_flag = false;
  sps.inter_entropy_continuation_enabled_flag = false;

  sps.seqBoundingBoxOrigin = 0;
  sps.seqBoundingBoxSize  = 0;
  sps.sps_bounding_box_offset_bits = 0;
  sps.sps_bounding_box_size_bits = 0;
}

static void
setupMinimalGPS(GeometryParameterSet& gps)
{
  gps.gps_geom_parameter_set_id = 0;
  gps.gps_seq_parameter_set_id  = 0;
  gps.predgeom_enabled_flag = true;

  // No duplicate-point merging in our simplified flow
  gps.geom_unique_points_flag = false;

  // No angular / azimuth scaling — pure Cartesian residual
  gps.geom_angular_mode_enabled_flag = false;
  gps.azimuth_scaling_enabled_flag   = false;
  gps.geom_angular_azimuth_scale_log2_minus11 = 0;
  gps.geom_angular_azimuth_speed_minus1 = 0;
  gps.geom_angular_radius_inv_scale_log2 = 0;
  gps.geom_slice_angular_origin_present_flag = false;
  gps.gpsAngularOrigin = 0;

  // No geometry scaling / QP
  gps.geom_scaling_enabled_flag = false;
  gps.geom_qp_multiplier_log2  = 0;
  gps.geom_base_qp = 0;
  gps.geom_idcm_qp_offset = 0;
  gps.geom_qp_offset_intvl_log2 = 0;

  // Prediction list
  gps.predgeom_max_pred_index = 0;
  gps.predgeom_radius_threshold_for_pred_list = 0;
  gps.resR_context_qphi_threshold = 0;
  gps.resR_context_qphi_threshold_present_flag = false;

  // Residual2 disabled (not relevant for flat loop)
  gps.residual2_disabled_flag = true;

  // No trisoup / octree / qtbt / inter
  gps.trisoup_enabled_flag    = false;
  gps.qtbt_enabled_flag       = false;
  gps.interPredictionEnabledFlag = false;
  gps.globalMotionEnabled     = false;
  gps.biPredictionEnabledFlag = 0;

  // Laser table — we'll set numLasers = 0 (no angular table)
  gps.angularTheta.clear();
  gps.angularZ.clear();
  gps.angularNumPhiPerTurn.clear();

  // Other flags that PredGeomEncoder reads
  gps.geom_box_log2_scale_present_flag = false;
  gps.gps_geom_box_log2_scale = 0;
  gps.neighbour_avail_boundary_log2_minus1 = 0;
  gps.inferred_direct_coding_mode = 0;
  gps.geom_planar_mode_enabled_flag = false;
  gps.bitwise_occupancy_coding_flag = true;
  gps.octree_point_count_list_present_flag = false;
}

static void
setupAngularGPS(GeometryParameterSet& gps, const SimpleParams& params)
{
  gps.geom_angular_mode_enabled_flag = true;
  gps.azimuth_scaling_enabled_flag = kAngularAzimuthScaling;
  gps.residual2_disabled_flag = kAngularResidual2Disabled;
  gps.resR_context_qphi_threshold_present_flag =
    kAngularQphiThresholdPresent;
  gps.resR_context_qphi_threshold = params.qphiThreshold;
  gps.geom_angular_azimuth_scale_log2_minus11 =
    params.azimuthScaleLog2;
  gps.geom_angular_radius_inv_scale_log2 =
    params.radiusInvScaleLog2;
  gps.geom_angular_azimuth_speed_minus1 =
    std::max(1, params.azimuthSpeed) - 1;
  gps.predgeom_max_pred_index                 = 3;
  gps.geom_slice_angular_origin_present_flag  = false;
  gps.gpsAngularOrigin = 0;

  // Match TMC3 --lasersTheta semantics:
  // angularTheta[i] = round(lasersTheta[i] * 2^18), where lasersTheta is the
  // configured per-laser value (commonly passed as the "theta" list used in
  // experiment scripts).
  static const double lasersThetaRaw[32] = {
    -0.53529, -0.51191, -0.48869, -0.46530, -0.44209, -0.41888, -0.39567, -0.37228,
    -0.34907, -0.32585, -0.30247, -0.27925, -0.25604, -0.23265, -0.20944, -0.18623,
    -0.16284, -0.13963, -0.11624, -0.09303, -0.06981, -0.04660, -0.02321,  0.00000,
     0.02321,  0.04660,  0.06981,  0.09303,  0.11641,  0.13963,  0.16284,  0.18623
  };
  gps.angularTheta.clear();
  gps.angularZ.clear();
  for (int i = 0; i < 32; i++) {
    gps.angularTheta.push_back(int(std::round(lasersThetaRaw[i] * double(1 << 18))));
    gps.angularZ.push_back(0);
  }
}

static void
setupMinimalGBH(GeometryBrickHeader& gbh)
{
  gbh.geom_geom_parameter_set_id = 0;
  gbh.geom_slice_id = 0;
  gbh.slice_tag = 0;
  gbh.frame_ctr_lsb = 0;
  gbh.prev_slice_id = 0;
  gbh.geomBoxOrigin = 0;
  gbh.gbhAngularOrigin = 0;
  gbh.geom_box_origin_bits_minus1 = 0;
  gbh.geom_box_log2_scale = 0;
  gbh.geom_slice_qp_offset = 0;
  gbh.geom_stream_cnt_minus1 = 0;       // single stream
  gbh.geom_qp_offset_intvl_log2_delta = 0;
  gbh.rootNodeSizeLog2 = 0;
  gbh.maxRootNodeDimLog2 = 0;
  gbh.entropy_continuation_flag = false;
  gbh.interPredictionEnabledFlag = false;
  gbh.biPredictionEnabledFlag = false;
  gbh.pgeom_min_radius = 0;

  // Residual bit-depth (set again inside encodePredictiveGeometry)
  for (int k = 0; k < 3; k++)
    gbh.pgeom_resid_abs_log2_bits[k] = 5;
}

static PredGeomEncOpts
makeMinimalPredGeomOpts()
{
  PredGeomEncOpts opt{};
  opt.sortMode = PredGeomEncOpts::kNoSort;
  opt.maxPtsPerTree = 0;
  opt.azimuthSortRecipBinWidth = 0;
  opt.maxPredIdxTested = 0;
  opt.radiusThresholdForNewPred = 0;
  opt.enablePartition = false;
  opt.splitter = 0;
  return opt;
}

static bool
isValidNumGroups(int g)
{
  return g == 1 || g == 2 || g == 4 || g == 8 || g == 16 || g == 32;
}

static bool
writeBitstreamFileTlv(
  const string& path,
  const PayloadBuffer& payload,
  bool angularMode)
{
  ofstream fout(path, ios::binary);
  if (!fout.is_open())
    return false;

  if (angularMode) {
    static const char kAngularMagic[4] = {'P', 'G', 'A', '2'};
    fout.write(kAngularMagic, sizeof(kAngularMagic));
    if (!fout.good())
      return false;
  }

  writeTlv(payload, fout);
  return fout.good();
}

static bool
readBitstreamFileTlv(const string& path, PayloadBuffer& payload, bool angularMode)
{
  ifstream fin(path, ios::binary);
  if (!fin.is_open())
    return false;

  if (angularMode) {
    static const char kAngularMagic[4] = {'P', 'G', 'A', '2'};
    char magic[sizeof(kAngularMagic)];
    fin.read(magic, sizeof(magic));
    if (!fin
        || !std::equal(
          std::begin(magic), std::end(magic), std::begin(kAngularMagic))) {
      cerr << "Error: angular bitstream syntax mismatch (expected PGA2 header)." << endl;
      return false;
    }
  }

  readTlv(fin, &payload);
  if (!fin || payload.type != PayloadType::kGeometryBrick)
    return false;
  return true;
}

static bool
writePointCloudPly(const PCCPointSet3& cloud, const double scale, const string& path)
{
  ply::PropertyNameMap names;
  names.position = {"x", "y", "z"};
  return ply::write(cloud, names, 1/scale, {0, 0, 0}, path, /*ascii=*/true);
}

static double
estimateAngularZeroRadiusRatio(
  const PCCPointSet3& cloud,
  int radiusInvScaleLog2)
{
  if (!cloud.getPointCount())
    return 0.0;

  size_t zeroCount = 0;
  for (size_t i = 0; i < cloud.getPointCount(); i++) {
    const auto& pt = cloud[i];
    const int64_t r0 = int64_t(std::round(std::hypot(double(pt[0]), double(pt[1]))));
    const int32_t sphR = int32_t(divExp2RoundHalfUp(r0, radiusInvScaleLog2));
    if (sphR == 0)
      zeroCount++;
  }

  return double(zeroCount) / double(cloud.getPointCount());
}

// Reorder points per-packet from firing-order (column-major) to laser-major.
//   original: f*numLasers + L        (f in [0,pointsPerLaser), L in [0,numLasers))
//   reorder : L*pointsPerLaser + f
// Trailing partial packet copied unchanged.
static void
reorderToLaserMajor(PCCPointSet3& cloud, int packetSize, int pointsPerLaser)
{
  if (packetSize <= 0 || pointsPerLaser <= 0
      || packetSize % pointsPerLaser != 0)
    return;
  const int numLasers = packetSize / pointsPerLaser;
  const size_t n = cloud.getPointCount();

  PCCPointSet3 out;
  out.addRemoveAttributes(cloud.hasColors(), cloud.hasReflectances());
  if (cloud.hasLaserAngles())
    out.addLaserAngles();
  out.resize(n);

  size_t p = 0;
  for (; p + size_t(packetSize) <= n; p += size_t(packetSize)) {
    for (int L = 0; L < numLasers; L++) {
      for (int f = 0; f < pointsPerLaser; f++) {
        const size_t src = p + size_t(f) * numLasers + L;
        const size_t dst = p + size_t(L) * pointsPerLaser + f;
        out[dst] = cloud[src];
        if (cloud.hasLaserAngles())
          out.setLaserAngle(dst, cloud.getLaserAngle(src));
        if (cloud.hasColors())
          out.setColor(dst, cloud.getColor(src));
        if (cloud.hasReflectances())
          out.setReflectance(dst, cloud.getReflectance(src));
      }
    }
  }
  for (; p < n; p++) {
    out[p] = cloud[p];
    if (cloud.hasLaserAngles())
      out.setLaserAngle(p, cloud.getLaserAngle(p));
    if (cloud.hasColors())
      out.setColor(p, cloud.getColor(p));
    if (cloud.hasReflectances())
      out.setReflectance(p, cloud.getReflectance(p));
  }
  cloud = std::move(out);
}

struct BppSummaryRow {
  string inputPlyPath;
  int numPoints = 0;
  int groups = 1;
  double inputScale = 1.0;
  string mode;
  int flatQp = -1;
  size_t payloadBytes = 0;
  double bpp = 0.0;
};

static bool
appendBppSummaryCsv(const string& path, const BppSummaryRow& row)
{
  const bool needsHeader = !ifstream(path).good();
  ofstream out(path, ios::app);
  if (!out.is_open())
    return false;

  if (needsHeader)
    out << "input,points,groups,scale,mode,flat_qp,payload_bytes,bpp\n";

  out << row.inputPlyPath << ','
      << row.numPoints << ','
      << row.groups << ','
      << std::setprecision(12) << row.inputScale << ','
      << row.mode << ','
      << row.flatQp << ','
      << row.payloadBytes << ','
      << std::setprecision(12) << row.bpp << '\n';

  return out.good();
}

static bool
parseBppSummaryCsvLine(const string& line, BppSummaryRow& row)
{
  if (line.empty())
    return false;

  vector<string> fields;
  string item;
  stringstream ss(line);
  while (getline(ss, item, ','))
    fields.push_back(item);

  if (fields.size() != 8)
    return false;

  try {
    row.inputPlyPath = fields[0];
    row.numPoints = stoi(fields[1]);
    row.groups = stoi(fields[2]);
    row.inputScale = stod(fields[3]);
    row.mode = fields[4];
    row.flatQp = stoi(fields[5]);
    row.payloadBytes = size_t(stoull(fields[6]));
    row.bpp = stod(fields[7]);
    return true;
  } catch (...) {
    return false;
  }
}

static bool
findLatestPairForInput(
  const string& path,
  const BppSummaryRow& key,
  BppSummaryRow* baselineQp0,
  BppSummaryRow* targetQp)
{
  if (!baselineQp0 || !targetQp)
    return false;

  *baselineQp0 = {};
  *targetQp = {};

  ifstream in(path);
  if (!in.is_open())
    return false;

  string line;
  bool firstLine = true;
  while (getline(in, line)) {
    if (firstLine) {
      firstLine = false;
      continue;
    }
    BppSummaryRow row;
    if (!parseBppSummaryCsvLine(line, row))
      continue;

    if (row.inputPlyPath != key.inputPlyPath
        || row.groups != key.groups
        || row.numPoints != key.numPoints
        || row.inputScale != key.inputScale)
      continue;

    if (row.mode.rfind("quantized_qp", 0) != 0)
      continue;

    if (row.flatQp == 0)
      *baselineQp0 = row;
    if (row.flatQp == key.flatQp)
      *targetQp = row;
  }

  return baselineQp0->payloadBytes > 0 && targetQp->payloadBytes > 0;
}

// ============================================================================
//  Main
// ============================================================================
int
main(int argc, char* argv[])
{
  cout << "=== TMC3 Simplified (PredGeom only) ===" << endl;

  // ---- 1. Parse options ----
  SimpleParams params;
  if (!parseSimpleArgs(argc, argv, params))
    return 1;
  if (!isValidNumGroups(params.numGroups)) {
    cerr << "Error: --groups must be one of {1,2,4,8,16,32}" << endl;
    return 1;
  }

  // ---- 2. Read PLY ----
  cout << "[1] Reading PLY: " << params.inputPlyPath << " ..." << endl;

  // Property names expected in the PLY (xyz + ring)
  ply::PropertyNameMap plyNames;
  plyNames.position = {"x", "y", "z"};

  PCCPointSet3 cloud;
  if (!ply::read(params.inputPlyPath, plyNames, params.inputScale, cloud)
      || cloud.getPointCount() == 0) {
    cerr << "Error: cannot open or empty input file!" << endl;
    return -1;
  }

  const auto numPoints = cloud.getPointCount();
  cout << "  Points loaded: " << numPoints
       << "  (inputScale=" << params.inputScale << ")" << endl;

  if (cloud.hasLaserAngles())
    cout << "  Using input laser/ring metadata from the PLY stream." << endl;
  else
    cout << "  No laser/ring metadata found; falling back to clamped point index." << endl;

  if (params.angularMode) {
    const double zeroRatio = estimateAngularZeroRadiusRatio(
      cloud, params.radiusInvScaleLog2);
    if (zeroRatio > 0.25) {
      cerr << "  Warning: " << std::fixed << std::setprecision(1)
           << (100.0 * zeroRatio)
           << "% of points map to sphR=0 at this --scale/--rate.\n"
           << "           This usually indicates a unit mismatch "
           << "(eg, meter-valued PLY with --scale 1).\n"
           << "           For nuScenes-style meter inputs, use --scale 1000."
           << std::defaultfloat << endl;
    }
  }

  if (params.batch && params.angularMode) {
    reorderToLaserMajor(cloud, kBatchPacketSize, kBatchPointsPerLaser);
    const int packets = int(cloud.getPointCount() / size_t(kBatchPacketSize));
    cout << "  Batch mode: laser-major within "
         << kBatchPacketSize << "-point packets ("
         << (kBatchPacketSize / kBatchPointsPerLaser) << " lasers x "
         << kBatchPointsPerLaser << " firings), "
         << packets << " full packets" << endl;
  }

  // ---- 3. Set up minimal parameter sets ----
  SequenceParameterSet sps;
  setupMinimalSPS(sps);

  GeometryParameterSet gps;
  setupMinimalGPS(gps);
  const int fixedQp = std::max(0, params.flatQp);

  GeometryBrickHeader gbh;
  setupMinimalGBH(gbh);

  PredGeomEncOpts predGeomOpt = makeMinimalPredGeomOpts();
  PredGeomContexts ctxtMem;

  // ---- 4. Encode ----
  // Wall + user timers
  pcc::chrono::Stopwatch<std::chrono::steady_clock> clock_wall;
  pcc::chrono::Stopwatch<pcc::chrono::utime_inc_children_clock> clock_user;

  if (params.angularMode) {
    const int angularPhiLog2 = params.azimuthScaleLog2 + 12;
    const int64_t angularScalePhi = int64_t(1) << angularPhiLog2;
    const int angularSpeed = std::max(1, params.azimuthSpeed);
    const int64_t angularMaxPtsPerRot =
      std::max<int64_t>(1, (angularScalePhi + (angularSpeed >> 1)) / angularSpeed);

    cout << "\n[2] Encoding (PredictiveGeometry, angular grid-based, spherical domain) ..." << endl;
    cout << "  scalePhi=" << angularScalePhi << "(2^" << angularPhiLog2
         << ")  azimuthSpeed=" << angularSpeed
         << "  maxPtsPerRot~" << angularMaxPtsPerRot
         << "  rings=32 (HDL-32E)" << endl;
    cout << "  positionRadiusInvScaleLog2="
         << params.radiusInvScaleLog2
         << "  predGeomAzimuthQuantization="
         << (kAngularAzimuthScaling ? 1 : 0)
         << "  secondaryResidualDisabled="
         << (kAngularResidual2Disabled ? 1 : 0) << endl;
    cout << "  resRContextQphiThresholdPresentFlag="
         << (kAngularQphiThresholdPresent ? 1 : 0)
         << "  resRContextQphiThreshold="
         << params.qphiThreshold << endl;
    cout << "  arcQuantLog2=" << params.arcLog2 << endl;
    cout << "  Syntax=predIdx + predictor-relative dRing/dColumn + residual_r + quantized_dAz" << endl;
  } else {
    cout << "\n[2] Encoding (PredictiveGeometry, flat ring-based loop, "
         << params.numGroups << " context group(s)) ..." << endl;
    cout << "  Flat residual coding mode: quantized-index" << endl;
    cout << "  Fixed flat QP (stream-signaled): " << fixedQp << endl;
    cout << "  Context toggles: range=" << (params.rangeCtxEnabled ? 1 : 0)
         << " boundary=" << (params.boundaryCtxEnabled ? 1 : 0) << endl;
    if (!params.dumpCsvPath.empty())
      cout << "  Dumping per-point context CSV to: " << params.dumpCsvPath << endl;
  }

  // Start payload buffer
  PayloadBuffer payload(PayloadType::kGeometryBrick);

  // Allocate arithmetic encoder
  int maxAcBufLen = int(numPoints) * 3 * 4 + 1024;
  std::unique_ptr<EntropyEncoder> aec(new EntropyEncoder(maxAcBufLen, nullptr));
  aec->enableBypassStream(sps.cabac_bypass_stream_enabled_flag);
  aec->setBypassBinCodingWithoutProbUpdate(sps.bypass_bin_coding_without_prob_update);
  aec->start();

  // --- Timed section ---
  clock_wall.start();
  clock_user.start();

  flat_predgeom::AngularMetrics metrics;
  if (params.angularMode) {
    setupAngularGPS(gps, params);
    encodePredictiveGeometryAngular(
      predGeomOpt, gps, gbh, cloud, ctxtMem, aec.get(),
      params.arcLog2, params.angularLossless,
      params.metrics ? &metrics : nullptr);
  } else {
    encodePredictiveGeometry(
      predGeomOpt, gps, gbh, cloud,
      ctxtMem, aec.get(), params.numGroups,
      fixedQp, params.rangeCtxEnabled, params.boundaryCtxEnabled, params.dumpCsvPath);
  }

  clock_user.stop();
  clock_wall.stop();

  // Signal the actual number of points coded
  gbh.footer.geom_num_points_minus1 = cloud.getPointCount() - 1;

  // ---- 5. Assemble payload buffer ----
  write(sps, gps, gbh, &payload);

  auto aecLen = aec->stop();
  auto aecBuf = aec->buffer();
  payload.insert(payload.end(), aecBuf, aecBuf + aecLen);
  write(gps, gbh, gbh.footer, &payload);

  // ---- 6. Write bitstream ----
  if (!writeBitstreamFileTlv(
        params.outputBitstreamPath, payload, params.angularMode)) {
    cerr << "Error: could not write bitstream file: " << params.outputBitstreamPath << endl;
    return 1;
  }
  cout << "\n[3] Bitstream written: " << params.outputBitstreamPath
       << " (" << payload.size() << " B)" << endl;

  // ---- 7. Write encoder reconstruction ----
  if (!writePointCloudPly(cloud, params.inputScale, params.reconstructedPath)) {
    cerr << "Error: could not write reconstructed PLY: " << params.reconstructedPath << endl;
    return 1;
  }
  cout << "[4] Encoder reconstruction written: " << params.reconstructedPath << endl;

  // ---- 8. Read bitstream file and decode ----
  cout << "\n[5] Decoding from bitstream file ..." << endl;

  PayloadBuffer payloadIn(PayloadType::kGeometryBrick);
  if (!readBitstreamFileTlv(
        params.outputBitstreamPath, payloadIn, params.angularMode)
      || payloadIn.empty()) {
    cerr << "Error: failed to read bitstream file: " << params.outputBitstreamPath << endl;
    return 1;
  }

  int bytesReadHead = 0;
  int bytesReadFoot = 0;
  auto gbhFromBitstream = parseGbh(sps, gps, payloadIn, &bytesReadHead, &bytesReadFoot);
  if (bytesReadHead < 0 || bytesReadFoot < 0
      || bytesReadHead + bytesReadFoot > int(payloadIn.size())) {
    cerr << "Error: invalid payload layout while parsing geometry brick." << endl;
    return 1;
  }

  const int codedGeomLen = int(payloadIn.size()) - bytesReadHead - bytesReadFoot;
  if (codedGeomLen <= 0) {
    cerr << "Error: geometry arithmetic payload is empty." << endl;
    return 1;
  }

  PCCPointSet3 decCloud;
  if (cloud.hasLaserAngles()) {
    decCloud.addLaserAngles();
    decCloud.resize(cloud.getPointCount());
    for (size_t i = 0; i < cloud.getPointCount(); i++)
      decCloud.setLaserAngle(i, cloud.getLaserAngle(i));
  }
  PredGeomContexts decCtxtMem;
  std::unique_ptr<EntropyDecoder> aed(new EntropyDecoder());
  aed->setBuffer(codedGeomLen, payloadIn.data() + bytesReadHead);
  aed->enableBypassStream(sps.cabac_bypass_stream_enabled_flag);
  aed->setBypassBinCodingWithoutProbUpdate(sps.bypass_bin_coding_without_prob_update);
  aed->start();

  if (params.angularMode) {
    setupAngularGPS(gps, params);
    decodePredictiveGeometryAngular(
      gps, gbhFromBitstream, decCloud, decCtxtMem, aed.get(),
      params.arcLog2, params.angularLossless);
  } else {
    decodePredictiveGeometry(
      gps, gbhFromBitstream, decCloud,
      decCtxtMem, aed.get(), params.numGroups,
      params.rangeCtxEnabled, params.boundaryCtxEnabled);
  }

  // ---- 9. Write decoder output ----
  if (!writePointCloudPly(decCloud, params.inputScale, params.decodedPath)) {
    cerr << "Error: could not write decoded PLY: " << params.decodedPath << endl;
    return 1;
  }
  cout << "[6] Decoded point cloud written: " << params.decodedPath << endl;

  // ---- 10. Verification ----
  cout << "\n[7] Verification ..." << endl;
  int mismatchCount = 0;
  if (decCloud.getPointCount() != cloud.getPointCount()) {
    cerr << "  Mismatch in point count! Encoded: " << cloud.getPointCount() 
         << ", Decoded: " << decCloud.getPointCount() << endl;
    mismatchCount++;
  } else {
    for (size_t i = 0; i < cloud.getPointCount(); i++) {
      if (cloud[i] != decCloud[i]) {
        if (mismatchCount < 10) {
          cerr << "  Mismatch at point " << i << ": "
               << "Enc=(" << cloud[i][0] << "," << cloud[i][1] << "," << cloud[i][2] << ") "
               << "Dec=(" << decCloud[i][0] << "," << decCloud[i][1] << "," << decCloud[i][2] << ")\n";
        }
        mismatchCount++;
      }
    }
  }

  if (mismatchCount == 0) {
    cout << "  Verification OK: all " << decCloud.getPointCount() << " reconstructed points match the encoded points!" << endl;
  } else {
    cout << "  Verification FAILED with " << mismatchCount << " mismatched points!" << endl;
  }

  // ---- 11. Report ----
  double bpp = double(8 * payload.size()) / numPoints;
  const string runMode = params.angularMode
    ? ("angular_s" + std::to_string(params.azimuthScaleLog2)
        + "_v" + std::to_string(std::max(1, params.azimuthSpeed))
        + "_r" + std::to_string(params.radiusInvScaleLog2)
        + "_qphi" + std::to_string(params.qphiThreshold)
        + (params.angularLossless
            ? "_lossless"
            : "_arc" + std::to_string(params.arcLog2)))
    : "quantized_qp" + std::to_string(fixedQp);
  cout << "\n[8] Results:" << endl;
  cout << "  Payload size:     " << payload.size() << " B ("
       << bpp << " bpp)" << endl;
  cout << "  Mode:             " << runMode << endl;

  if (params.metrics && params.angularMode) {
    const double sameRingRate = double(metrics.sameRingHit) / double(numPoints);
    cout << "  Metrics:" << endl;
    cout << "    predIdx hist   : [" << metrics.predIdxHist[0]
         << "," << metrics.predIdxHist[1]
         << "," << metrics.predIdxHist[2]
         << "," << metrics.predIdxHist[3] << "]" << endl;
    cout << "    sameRingHit    : " << metrics.sameRingHit << " / "
         << numPoints << " (" << sameRingRate << ")" << endl;
    cout << "    ringTransitions: " << metrics.ringTransitions << endl;
  }

  BppSummaryRow runRow;
  runRow.inputPlyPath = params.inputPlyPath;
  runRow.numPoints = int(numPoints);
  runRow.groups = params.numGroups;
  runRow.inputScale = params.inputScale;
  runRow.flatQp = params.angularMode ? -1 : fixedQp;
  runRow.mode = runMode;
  runRow.payloadBytes = payload.size();
  runRow.bpp = bpp;

  if (appendBppSummaryCsv(params.summaryCsvPath, runRow)) {
    cout << "  Summary CSV:      " << params.summaryCsvPath << endl;
    BppSummaryRow qp0Row;
    BppSummaryRow currentRow;
    if (!params.angularMode && fixedQp > 0
        && findLatestPairForInput(params.summaryCsvPath, runRow, &qp0Row, &currentRow)) {
      const double dBytes = double(currentRow.payloadBytes) - double(qp0Row.payloadBytes);
      const double dBpp = currentRow.bpp - qp0Row.bpp;
      const double dPct = qp0Row.bpp > 0 ? (100.0 * dBpp / qp0Row.bpp) : 0.0;
      cout << "\n  Paired bpp comparison vs QP0:" << endl;
      cout << "    qp0      : " << qp0Row.payloadBytes << " B, " << qp0Row.bpp << " bpp" << endl;
      cout << "    qp" << currentRow.flatQp << "      : "
           << currentRow.payloadBytes << " B, " << currentRow.bpp << " bpp" << endl;
      cout << "    delta    : " << dBytes << " B, " << dBpp << " bpp ("
           << std::showpos << dPct << "%" << std::noshowpos << ")" << endl;
    } else if (!params.angularMode && fixedQp == 0) {
      cout << "  QP0 run recorded (reference point for paired comparisons)." << endl;
    } else if (params.angularMode) {
      cout << "  Angular run recorded." << endl;
    } else {
      cout << "  Paired comparison not available yet "
           << "(need a QP0 row for this input)." << endl;
    }
  } else {
    cout << "  Warning: failed to append summary CSV at " << params.summaryCsvPath << endl;
  }

  using namespace std::chrono;
  auto total_wall = duration_cast<milliseconds>(clock_wall.count()).count();
  auto total_user = duration_cast<milliseconds>(clock_user.count()).count();
  cout << "  Processing time (wall): " << total_wall / 1000.0 << " s" << endl;
  cout << "  Processing time (user): " << total_user / 1000.0 << " s" << endl;

  cout << "\nDone." << endl;
  return 0;
}
