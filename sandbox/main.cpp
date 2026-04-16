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

#include <cassert>
#include <chrono>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <memory>
#include <sstream>
#include <string>
#include <vector>

#include "PayloadBuffer.h"
#include "entropy.h"
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
  string inputPlyPath = "/home/swnh/pgc/datasets/nuscenes/v1.0-mini/ply/bin/scene-0061/scene-0061_00.ply";  // --input
  string outputBitstreamPath = "/home/swnh/gpcc/experiments/predgeom.bin";  // --output
  string reconstructedPath = "/home/swnh/gpcc/experiments/recon.ply";       // --recon
  string decodedPath = "/home/swnh/gpcc/experiments/decoded.ply";           // --decoded
  string dumpCsvPath;                           // --dump-csv
  string summaryCsvPath = "scripts/flat_qres_qp_summary.csv";  // --summary-csv
  double inputScale = 1.0;                      // --scale
  int    numGroups  = 1;                        // --groups
  int    flatQp = -1;                           // --flat-qp
  bool   rangeCtxEnabled = true;                // --range-ctx
  bool   boundaryCtxEnabled = true;             // --boundary-ctx
  bool   angularMode = false;                   // --angular
};

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
  double qpInt);

void decodePredictiveGeometryAngular(
  const GeometryParameterSet&,
  const GeometryBrickHeader&,
  PCCPointSet3&,
  PredGeomContexts&,
  EntropyDecoder*,
  double qpInt);

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
static bool
parseSimpleArgs(int argc, char* argv[], SimpleParams& p)
{
  if (argc < 3) {
    cerr << "Usage: " << argv[0]
         << " --input <ply> [--output <bitstream.bin>] [--recon <recon.ply>]"
         << " [--decoded <decoded.ply>]"
         << " [--scale <inputScale>] [--groups <1|2|4|8|16|32>]"
         << " [--dump-csv <context.csv>]"
         << " [--flat-qp <int>] [--range <0|1>] [--boundary <0|1>]"
         << " [--summary-csv <summary.csv>] [--angular]\n";
    return false;
  }
  for (int i = 1; i < argc; i++) {
    string arg = argv[i];
    if ((arg == "--input" || arg == "-i") && i + 1 < argc)
      p.inputPlyPath = argv[++i];
    else if ((arg == "--output" || arg == "-o") && i + 1 < argc)
      p.outputBitstreamPath = argv[++i];
    else if ((arg == "--recon" || arg == "-r") && i + 1 < argc)
      p.reconstructedPath = argv[++i];
    else if ((arg == "--decoded" || arg == "-d") && i + 1 < argc)
      p.decodedPath = argv[++i];
    else if ((arg == "--scale" || arg == "-s") && i + 1 < argc)
      p.inputScale = atof(argv[++i]);
    else if ((arg == "--groups" || arg == "-g") && i + 1 < argc)
      p.numGroups = atoi(argv[++i]);
    else if (arg == "--dump-csv" && i + 1 < argc)
      p.dumpCsvPath = argv[++i];
    else if (arg == "--summary-csv" && i + 1 < argc)
      p.summaryCsvPath = argv[++i];
    else if ((arg == "--flat-qp" || arg == "--qp") && i + 1 < argc)
      p.flatQp = atoi(argv[++i]);
    else if (arg == "--range" && i + 1 < argc)
      p.rangeCtxEnabled = atoi(argv[++i]) != 0;
    else if (arg == "--boundary" && i + 1 < argc)
      p.boundaryCtxEnabled = atoi(argv[++i]) != 0;
    else if (arg == "--angular")
      p.angularMode = true;
    else {
      cerr << "Unknown option: " << arg << "\n";
      return false;
    }
  }
  if (p.inputPlyPath.empty()) {
    cerr << "Error: --input is required\n";
    return false;
  }
  if (p.flatQp < -1) {
    cerr << "Error: --flat-qp must be >= 0 when provided\n";
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
setupAngularGPS(GeometryParameterSet& gps)
{
  gps.geom_angular_mode_enabled_flag = true;
  gps.azimuth_scaling_enabled_flag   = false;
  gps.geom_angular_azimuth_scale_log2_minus11 = 7;  // scalePhi = 2^19 = 524288
  gps.geom_angular_radius_inv_scale_log2      = 0;  // sphR = raw internal units (mm)
  gps.geom_angular_azimuth_speed_minus1       = 0;
  gps.geom_slice_angular_origin_present_flag  = false;
  gps.gpsAngularOrigin = 0;

  // Velodyne HDL-32E: 32 rings, -30.67° to +10.67°, ~1.33° steps
  // angularTheta[i] = round(tan(elev_i * pi/180) * 2^18)
  static const double elevDeg[32] = {
    -30.67, -29.33, -28.00, -26.67, -25.33, -24.00, -22.67, -21.33,
    -20.00, -18.67, -17.33, -16.00, -14.67, -13.33, -12.00, -10.67,
     -9.33,  -8.00,  -6.67,  -5.33,  -4.00,  -2.67,  -1.33,   0.00,
      1.33,   2.67,   4.00,   5.33,   6.67,   8.00,   9.33,  10.67
  };
  gps.angularTheta.clear();
  gps.angularZ.clear();
  for (int i = 0; i < 32; i++) {
    gps.angularTheta.push_back(
      int(std::round(std::tan(elevDeg[i] * M_PI / 180.0) * double(1 << 18))));
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
writeBitstreamFileTlv(const string& path, const PayloadBuffer& payload)
{
  ofstream fout(path, ios::binary);
  if (!fout.is_open())
    return false;
  writeTlv(payload, fout);
  return fout.good();
}

static bool
readBitstreamFileTlv(const string& path, PayloadBuffer& payload)
{
  ifstream fin(path, ios::binary);
  if (!fin.is_open())
    return false;
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

  // Angular QP: 2^flat_qp mm (flat_qp=-1 → default 64 mm = 2^6)
  const double angQp = (params.flatQp >= 0) ? double(1 << params.flatQp) : 64.0;

  if (params.angularMode) {
    cout << "\n[2] Encoding (PredictiveGeometry, angular grid-based, spherical domain) ..." << endl;
    cout << "  scalePhi=524288(2^19)  maxPtsPerRot=1090  rings=32 (HDL-32E)" << endl;
    cout << "  QP=" << angQp << " mm  (--flat-qp " << params.flatQp << " → 2^" << params.flatQp << ")" << endl;
    cout << "  Occupancy=(dRing,dColumn) + predIdx + quantized_dr + quantized_dAz" << endl;
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

  if (params.angularMode) {
    setupAngularGPS(gps);
    encodePredictiveGeometryAngular(predGeomOpt, gps, gbh, cloud, ctxtMem, aec.get(), angQp);
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
  if (!writeBitstreamFileTlv(params.outputBitstreamPath, payload)) {
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
  if (!readBitstreamFileTlv(params.outputBitstreamPath, payloadIn) || payloadIn.empty()) {
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
    setupAngularGPS(gps);
    decodePredictiveGeometryAngular(gps, gbhFromBitstream, decCloud, decCtxtMem, aed.get(), angQp);
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
  cout << "\n[8] Results:" << endl;
  cout << "  Payload size:     " << payload.size() << " B ("
       << bpp << " bpp)" << endl;
  cout << "  Mode:             quantized_qp" << fixedQp << endl;

  BppSummaryRow runRow;
  runRow.inputPlyPath = params.inputPlyPath;
  runRow.numPoints = int(numPoints);
  runRow.groups = params.numGroups;
  runRow.inputScale = params.inputScale;
  runRow.flatQp = fixedQp;
  runRow.mode = "quantized_qp" + std::to_string(fixedQp);
  runRow.payloadBytes = payload.size();
  runRow.bpp = bpp;

  if (appendBppSummaryCsv(params.summaryCsvPath, runRow)) {
    cout << "  Summary CSV:      " << params.summaryCsvPath << endl;
    BppSummaryRow qp0Row;
    BppSummaryRow currentRow;
    if (fixedQp > 0
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
    } else if (fixedQp == 0) {
      cout << "  QP0 run recorded (reference point for paired comparisons)." << endl;
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
