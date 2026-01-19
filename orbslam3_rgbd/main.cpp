#include "calibration.hpp"

#include <System.h>
#include <Tracking.h>

#include <opencv2/opencv.hpp>

#include <rerun.hpp>
#include <rerun/archetypes/pinhole.hpp>

#include <Eigen/Core>

#include <algorithm>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <memory>
#include <sstream>
#include <string>
#include <vector>
#include <array>
#include <utility>

namespace fs = std::filesystem;
namespace rr = rerun;

struct Options {
  std::string rgb_dir;
  std::string depth_dir;
  std::string calib_path;
  std::string timestamps_path;
  std::string vocab_path;
  std::string out_dir;
  std::string out_path;
  std::string keyframe_path;
  std::string rgb_ext = ".jpg";
  std::string depth_ext = ".png";
  double depth_scale = 1000.0;
  double balance = 0.0;
  bool use_projection = true;
  bool undistort = true;
  std::string distortion_model;
  double camera_fps = 0.0;
  bool viewer = false;
  bool save_keyframes = false;
  int max_frames = -1;
  bool skip_missing = false;
  bool rerun_enabled = false;
  bool rerun_spawn = false;
  std::string rerun_addr = "rerun+http://127.0.0.1:9876/proxy";
};

struct FrameEntry {
  std::string frame_id;
  double timestamp_sec;
};

static std::string default_vocab_path()
{
  const fs::path root = fs::path(__FILE__).parent_path().parent_path();
  const fs::path vocab = root / "third_party" / "ORB_SLAM3" / "Vocabulary" / "ORBvoc.txt";
  if (fs::exists(vocab)) {
    return vocab.string();
  }
  const fs::path fallback = fs::path(__FILE__).parent_path() / "vocabulary" / "ORBvoc.txt";
  return fallback.string();
}

static std::string normalize_ext(const std::string & ext)
{
  if (ext.empty()) {
    return ext;
  }
  if (ext.front() == '.') {
    return ext;
  }
  return "." + ext;
}

static void print_usage(const char * prog)
{
  std::cout << "Usage: " << prog << " --rgb-dir DIR --depth-dir DIR --timestamps PATH --calibration PATH [options]\n"
            << "  --rgb-dir DIR          Folder containing RGB images\n"
            << "  --depth-dir DIR        Folder containing depth images\n"
            << "  --timestamps PATH      timestamps.txt (frame,timestamp_ns)\n"
            << "  --calibration PATH     Camera calibration YAML (K/D or camera_info)\n"
            << "  --vocab PATH           ORB-SLAM3 vocabulary file\n"
            << "  --out-dir DIR          Output directory (default: parent of rgb-dir)\n"
            << "  --out PATH             Output trajectory file (TUM format)\n"
            << "  --save-keyframes       Save keyframe trajectory (TUM format)\n"
            << "  --rgb-ext EXT          RGB extension (default: .jpg)\n"
            << "  --depth-ext EXT        Depth extension (default: .png)\n"
            << "  --depth-scale VALUE    Depth scale factor (default: 1000)\n"
            << "  --undistort            Undistort images before tracking (default)\n"
            << "  --no-undistort         Disable undistortion\n"
            << "  --distortion-model M   brown|fisheye|equidistant (override)\n"
            << "  --balance VALUE        Undistort balance (0..1)\n"
            << "  --no-projection        Ignore projection matrix if present\n"
            << "  --camera-fps VALUE     Camera FPS for ORB-SLAM3 config\n"
            << "  --viewer               Enable Pangolin viewer\n"
            << "  --max-frames N         Limit number of frames\n"
            << "  --skip-missing         Skip missing frames instead of failing\n"
            << "  --rerun                Enable Rerun logging\n"
            << "  --rerun-spawn          Spawn a Rerun viewer (otherwise connect to --rerun-addr)\n"
            << "  --rerun-addr URL       Rerun gRPC URL (default: rerun+http://127.0.0.1:9876/proxy)\n"
            << "  -h, --help             Show this message\n";
}

static Options parse_args(int argc, char ** argv)
{
  Options opt;
  opt.vocab_path = default_vocab_path();

  for (int i = 1; i < argc; ++i) {
    const std::string arg = argv[i];
    auto next = [&]() -> std::string {
      if (i + 1 >= argc) {
        throw std::runtime_error("Missing value for " + arg);
      }
      return argv[++i];
    };

    if (arg == "--rgb-dir") {
      opt.rgb_dir = next();
    } else if (arg == "--depth-dir") {
      opt.depth_dir = next();
    } else if (arg == "--timestamps") {
      opt.timestamps_path = next();
    } else if (arg == "--calibration") {
      opt.calib_path = next();
    } else if (arg == "--vocab") {
      opt.vocab_path = next();
    } else if (arg == "--out-dir") {
      opt.out_dir = next();
    } else if (arg == "--out") {
      opt.out_path = next();
    } else if (arg == "--save-keyframes") {
      opt.save_keyframes = true;
    } else if (arg == "--rgb-ext") {
      opt.rgb_ext = normalize_ext(next());
    } else if (arg == "--depth-ext") {
      opt.depth_ext = normalize_ext(next());
    } else if (arg == "--depth-scale") {
      opt.depth_scale = std::stod(next());
    } else if (arg == "--undistort") {
      opt.undistort = true;
    } else if (arg == "--no-undistort") {
      opt.undistort = false;
    } else if (arg == "--distortion-model") {
      opt.distortion_model = next();
    } else if (arg == "--balance") {
      opt.balance = std::stod(next());
    } else if (arg == "--no-projection") {
      opt.use_projection = false;
    } else if (arg == "--camera-fps") {
      opt.camera_fps = std::stod(next());
    } else if (arg == "--viewer") {
      opt.viewer = true;
    } else if (arg == "--max-frames") {
      opt.max_frames = std::stoi(next());
    } else if (arg == "--skip-missing") {
      opt.skip_missing = true;
    } else if (arg == "--rerun") {
      opt.rerun_enabled = true;
    } else if (arg == "--rerun-spawn") {
      opt.rerun_enabled = true;
      opt.rerun_spawn = true;
    } else if (arg == "--rerun-addr") {
      opt.rerun_enabled = true;
      opt.rerun_addr = next();
    } else if (arg == "-h" || arg == "--help") {
      print_usage(argv[0]);
      std::exit(0);
    } else {
      throw std::runtime_error("Unknown option: " + arg);
    }
  }

  if (opt.rgb_dir.empty() || opt.depth_dir.empty() || opt.timestamps_path.empty() || opt.calib_path.empty()) {
    throw std::runtime_error("Missing required arguments.");
  }

  if (opt.out_dir.empty()) {
    opt.out_dir = fs::path(opt.rgb_dir).parent_path().string();
  }
  if (opt.out_path.empty()) {
    opt.out_path = (fs::path(opt.out_dir) / "orbslam3_poses.tum").string();
  }
  if (opt.save_keyframes && opt.keyframe_path.empty()) {
    opt.keyframe_path = (fs::path(opt.out_dir) / "orbslam3_keyframes.tum").string();
  }

  return opt;
}

static std::vector<FrameEntry> read_timestamps(const std::string & path)
{
  std::vector<FrameEntry> entries;
  std::ifstream file(path);
  if (!file.is_open()) {
    return entries;
  }

  std::string line;
  while (std::getline(file, line)) {
    if (line.empty()) {
      continue;
    }
    if (line.find("frame") != std::string::npos && line.find("timestamp") != std::string::npos) {
      continue;
    }

    std::string frame;
    std::string ts_str;
    const auto comma = line.find(',');
    if (comma != std::string::npos) {
      frame = line.substr(0, comma);
      ts_str = line.substr(comma + 1);
    } else {
      std::istringstream iss(line);
      iss >> frame >> ts_str;
    }
    if (frame.empty() || ts_str.empty()) {
      continue;
    }
    try {
      const long long ts_ns = std::stoll(ts_str);
      entries.push_back({frame, static_cast<double>(ts_ns) / 1e9});
    } catch (...) {
    }
  }
  return entries;
}

static double estimate_fps(const std::vector<FrameEntry> & frames, double fallback)
{
  if (frames.size() < 2) {
    return fallback;
  }
  double total = 0.0;
  int count = 0;
  for (size_t i = 1; i < frames.size(); ++i) {
    const double dt = frames[i].timestamp_sec - frames[i - 1].timestamp_sec;
    if (dt > 0.0) {
      total += dt;
      ++count;
    }
  }
  if (count == 0) {
    return fallback;
  }
  return 1.0 / (total / static_cast<double>(count));
}

static bool write_orbslam_config(
  const std::string & path,
  const cv::Mat & k,
  int width,
  int height,
  double fps,
  double depth_scale)
{
  std::ofstream file(path);
  if (!file.is_open()) {
    return false;
  }

  int fps_int = static_cast<int>(std::lround(fps));
  if (fps_int <= 0) {
    fps_int = 30;
  }

  file << "%YAML:1.0\n\n";
  file << "File.version: \"1.0\"\n\n";
  file << "Camera.type: \"PinHole\"\n\n";
  file << "Camera1.fx: " << k.at<double>(0, 0) << "\n";
  file << "Camera1.fy: " << k.at<double>(1, 1) << "\n";
  file << "Camera1.cx: " << k.at<double>(0, 2) << "\n";
  file << "Camera1.cy: " << k.at<double>(1, 2) << "\n\n";
  file << "Camera1.k1: 0.0\n";
  file << "Camera1.k2: 0.0\n";
  file << "Camera1.p1: 0.0\n";
  file << "Camera1.p2: 0.0\n";
  file << "Camera1.k3: 0.0\n\n";
  file << "Camera.width: " << width << "\n";
  file << "Camera.height: " << height << "\n\n";
  file << "Camera.newWidth: " << width << "\n";
  file << "Camera.newHeight: " << height << "\n\n";
  file << "Camera.fps: " << fps_int << "\n";
  file << "Camera.RGB: 0\n\n";
  file << "Stereo.ThDepth: 40.0\n";
  file << "Stereo.b: 0.0\n\n";
  file << "RGBD.DepthMapFactor: " << std::fixed << std::setprecision(6) << depth_scale << "\n\n";
  file << "ORBextractor.nFeatures: 1000\n";
  file << "ORBextractor.scaleFactor: 1.2\n";
  file << "ORBextractor.nLevels: 8\n";
  file << "ORBextractor.iniThFAST: 20\n";
  file << "ORBextractor.minThFAST: 7\n";
  file << "\n";
  file << "Viewer.KeyFrameSize: 0.05\n";
  file << "Viewer.KeyFrameLineWidth: 1.0\n";
  file << "Viewer.GraphLineWidth: 0.9\n";
  file << "Viewer.PointSize: 2.0\n";
  file << "Viewer.CameraSize: 0.08\n";
  file << "Viewer.CameraLineWidth: 3.0\n";
  file << "Viewer.ViewpointX: 0.0\n";
  file << "Viewer.ViewpointY: -0.7\n";
  file << "Viewer.ViewpointZ: -1.8\n";
  file << "Viewer.ViewpointF: 500.0\n";
  return true;
}

int main(int argc, char ** argv)
{
  Options opt;
  try {
    opt = parse_args(argc, argv);
  } catch (const std::exception & e) {
    std::cerr << e.what() << "\n";
    print_usage(argv[0]);
    return 1;
  }

  if (!fs::exists(opt.vocab_path)) {
    std::cerr << "ORB vocabulary not found: " << opt.vocab_path << "\n";
    return 1;
  }

  std::vector<FrameEntry> frames = read_timestamps(opt.timestamps_path);
  if (frames.empty()) {
    std::cerr << "No timestamps found in " << opt.timestamps_path << "\n";
    return 1;
  }

  const fs::path first_rgb = fs::path(opt.rgb_dir) / (frames.front().frame_id + opt.rgb_ext);
  if (!fs::exists(first_rgb)) {
    std::cerr << "First RGB frame not found: " << first_rgb << "\n";
    return 1;
  }

  cv::Mat first_image = cv::imread(first_rgb.string(), cv::IMREAD_COLOR);
  if (first_image.empty()) {
    std::cerr << "Failed to read: " << first_rgb << "\n";
    return 1;
  }

  recons::CalibrationData calib;
  if (!recons::LoadCalibrationYaml(opt.calib_path, calib)) {
    std::cerr << "Failed to load calibration from " << opt.calib_path << "\n";
    return 1;
  }

  if (calib.width <= 0 || calib.height <= 0) {
    calib.width = first_image.cols;
    calib.height = first_image.rows;
  }

  cv::Mat map1, map2, new_k;
  if (opt.undistort) {
    if (!recons::BuildUndistortMaps(calib, opt.balance, opt.use_projection, opt.distortion_model, map1, map2, new_k)) {
      std::cerr << "Failed to build undistort maps.\n";
      return 1;
    }
  } else {
    new_k = (cv::Mat_<double>(3, 3) <<
      calib.k[0], calib.k[1], calib.k[2],
      calib.k[3], calib.k[4], calib.k[5],
      calib.k[6], calib.k[7], calib.k[8]);
  }

  if (opt.camera_fps <= 0.0) {
    opt.camera_fps = estimate_fps(frames, 30.0);
  }

  // Rerun setup (optional)
  std::unique_ptr<rr::RecordingStream> rerun_stream;
  if (opt.rerun_enabled) {
    rerun_stream = std::make_unique<rr::RecordingStream>("ORB-SLAM3 RGBD");
    rr::Error err;
    if (opt.rerun_spawn) {
      err = rerun_stream->spawn();
    } else {
      err = rerun_stream->connect_grpc(opt.rerun_addr);
    }
    if (err.is_err()) {
      std::cerr << "Failed to start Rerun stream: " << err.description << "\n";
      rerun_stream.reset();
    } else {
      rerun_stream->set_global();
      const float fx = static_cast<float>(new_k.at<double>(0, 0));
      const float fy = static_cast<float>(new_k.at<double>(1, 1));
      const float cx = static_cast<float>(new_k.at<double>(0, 2));
      const float cy = static_cast<float>(new_k.at<double>(1, 2));
      (void)cx;
      (void)cy;
      auto pinhole = rr::Pinhole::from_focal_length_and_resolution(
        {fx, fy},
        {static_cast<float>(calib.width), static_cast<float>(calib.height)}
      );
      rerun_stream->log_static("world/camera", pinhole);
    }
  }

  fs::create_directories(opt.out_dir);
  const fs::path config_path = fs::path(opt.out_dir) / "orbslam3_runtime.yaml";
  if (!write_orbslam_config(config_path.string(), new_k, calib.width, calib.height, opt.camera_fps, opt.depth_scale)) {
    std::cerr << "Failed to write ORB-SLAM3 config.\n";
    return 1;
  }

  std::cout << "ORB-SLAM3 config: " << config_path << "\n"
            << "Vocabulary: " << opt.vocab_path << "\n"
            << "Calibration: " << opt.calib_path << "\n"
            << "Output: " << opt.out_path << "\n";

  ORB_SLAM3::System slam(opt.vocab_path, config_path.string(), ORB_SLAM3::System::RGBD, opt.viewer);

  std::vector<rr::datatypes::Vec3D> trajectory;

  int processed = 0;
  int tracked = 0;
  for (const auto & entry : frames) {
    if (opt.max_frames > 0 && processed >= opt.max_frames) {
      break;
    }

    const fs::path rgb_path = fs::path(opt.rgb_dir) / (entry.frame_id + opt.rgb_ext);
    const fs::path depth_path = fs::path(opt.depth_dir) / (entry.frame_id + opt.depth_ext);
    if (!fs::exists(rgb_path) || !fs::exists(depth_path)) {
      if (opt.skip_missing) {
        ++processed;
        continue;
      }
      std::cerr << "Missing frame: " << rgb_path << " or " << depth_path << "\n";
      return 1;
    }

    cv::Mat rgb = cv::imread(rgb_path.string(), cv::IMREAD_COLOR);
    cv::Mat depth = cv::imread(depth_path.string(), cv::IMREAD_UNCHANGED);
    if (rgb.empty() || depth.empty()) {
      if (opt.skip_missing) {
        ++processed;
        continue;
      }
      std::cerr << "Failed to read: " << rgb_path << " or " << depth_path << "\n";
      return 1;
    }
    if (depth.type() != CV_16U && depth.type() != CV_32F) {
      depth.convertTo(depth, CV_32F);
    }

    if (opt.undistort) {
      cv::Mat rgb_rect;
      cv::Mat depth_rect;
      cv::remap(rgb, rgb_rect, map1, map2, cv::INTER_LINEAR, cv::BORDER_CONSTANT);
      cv::remap(depth, depth_rect, map1, map2, cv::INTER_NEAREST, cv::BORDER_CONSTANT, 0);
      rgb = rgb_rect;
      depth = depth_rect;
    }

    const Sophus::SE3f Tcw = slam.TrackRGBD(rgb, depth, entry.timestamp_sec);
    const int tracking_state = slam.GetTrackingState();
    const bool tracking_ok =
      tracking_state == ORB_SLAM3::Tracking::OK ||
      tracking_state == ORB_SLAM3::Tracking::OK_KLT;
    if (tracking_ok) {
      ++tracked;
    }

    if (rerun_stream) {
      rerun_stream->set_time_timestamp_secs_since_epoch("time", entry.timestamp_sec);

      cv::Mat rgb_rgb;
      cv::cvtColor(rgb, rgb_rgb, cv::COLOR_BGR2RGB);
      rerun_stream->log(
        "world/camera/image",
        rr::Image(
          rr::Collection<uint8_t>::borrow(
            rgb_rgb.data, static_cast<size_t>(rgb_rgb.total() * rgb_rgb.channels())),
          {static_cast<uint32_t>(rgb_rgb.cols), static_cast<uint32_t>(rgb_rgb.rows)},
          rr::datatypes::ColorModel::RGB,
          rr::datatypes::ChannelDatatype::U8)
      );

      cv::Mat depth_m;
      if (depth.type() == CV_16U) {
        depth.convertTo(depth_m, CV_32F, 1.0 / opt.depth_scale);
      } else {
        depth.convertTo(depth_m, CV_32F);
      }
      rerun_stream->log(
        "world/camera/depth",
        rr::DepthImage(
          depth_m.data,
          {static_cast<uint32_t>(depth_m.cols), static_cast<uint32_t>(depth_m.rows)},
          rr::datatypes::ChannelDatatype::F32)
      );

      if (tracking_ok) {
        const Eigen::Matrix4f Twc = Tcw.inverse().matrix();
        const Eigen::Vector3f t = Twc.block<3, 1>(0, 3);
        Eigen::Quaternionf q(Twc.block<3, 3>(0, 0));
        rerun_stream->log(
          "world/camera/transform",
          rr::Transform3D()
            .with_translation({t.x(), t.y(), t.z()})
            .with_rotation(rr::datatypes::Quaternion{q.x(), q.y(), q.z(), q.w()})
        );
        trajectory.emplace_back(t.x(), t.y(), t.z());
        // Copy trajectory into an owned collection to avoid lifetime issues when it grows.
        auto traj_copy = trajectory;
        rr::components::LineStrip3D strip(rr::Collection<rr::datatypes::Vec3D>::take_ownership(std::move(traj_copy)));
        rerun_stream->log(
          "world/camera/path",
          rr::LineStrips3D(rr::Collection<rr::components::LineStrip3D>::take_ownership(std::move(strip)))
        );
      }
    }

    ++processed;
    if (processed % 50 == 0) {
      std::cout << "Processed " << processed << " frames, tracked " << tracked << "\n";
    }
  }

  slam.Shutdown();
  slam.SaveTrajectoryTUM(opt.out_path);
  if (opt.save_keyframes) {
    slam.SaveKeyFrameTrajectoryTUM(opt.keyframe_path);
  }

  std::cout << "Done. Tracked " << tracked << "/" << processed << " frames.\n";
  std::cout << "Saved poses to " << opt.out_path << "\n";
  if (opt.save_keyframes) {
    std::cout << "Saved keyframes to " << opt.keyframe_path << "\n";
  }
  return 0;
}
