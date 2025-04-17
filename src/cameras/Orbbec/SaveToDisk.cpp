#include <cmath>
#include <fstream>
#include <iostream>
#include <opencv2/opencv.hpp>
#include <vector>
#include <sstream>
#include <iomanip>  // For setw and setfill

#include "libobsensor/ObSensor.hpp"
#include "utils.hpp"

using namespace std;

#define KEY_ESC 27
#define KEY_R 82
#define KEY_r 114
#define KEY_n 110
#define KEY_N 78
#define KEY_p 112
#define KEY_s 115

#include <chrono>
#include <condition_variable>
#include <cstring>
#include <exception>
#include <mutex>
#include <string>
#include <thread>

#include "libobsensor/hpp/Error.hpp"

bool isWaitRebootComplete_ = false;
bool isDeviceRemoved_ = false;
std::condition_variable waitRebootCondition_;
std::mutex waitRebootMutex_;
std::string deviceUid_;
std::string deviceSN_;
std::shared_ptr<ob::Device> rebootedDevice_;

int getch_noblock() {
  if (kbhit())
      return getch();
  else
      return -1;
}

void saveMetaDatatoFile(OBCameraParam &cameraParam, std::ofstream &metaFile) {
  metaFile << "Camera Intrinsics (Depth):\n";
  metaFile << "fx: " << cameraParam.depthIntrinsic.fx << "\n";
  metaFile << "fy: " << cameraParam.depthIntrinsic.fy << "\n";
  metaFile << "cx: " << cameraParam.depthIntrinsic.cx << "\n";
  metaFile << "cy: " << cameraParam.depthIntrinsic.cy << "\n";
  metaFile << "Camera Distortion (Depth):\n";
  metaFile << "k1: " << cameraParam.depthDistortion.k1 << "\n";
  metaFile << "k2: " << cameraParam.depthDistortion.k2 << "\n";
  metaFile << "p1: " << cameraParam.depthDistortion.p1 << "\n";
  metaFile << "p2: " << cameraParam.depthDistortion.p2 << "\n";
  metaFile << "SDK version: " << ob::Version::getMajor() << "."
           << ob::Version::getMinor() << "." << ob::Version::getPatch()
           << std::endl;
  metaFile << "SDK stage version: " << ob::Version::getStageVersion()
           << std::endl;
}
// Print device name, serial number, vid, pid and firmware version.
void dumpDeviceInfo(std::shared_ptr<ob::Device> device,
                    std::ofstream &metafile) {
  // Get device information
  auto devInfo = device->getDeviceInfo();

  // Get the name of the device
  metafile << "Device name: " << devInfo->name() << std::endl;

  // Get the pid, vid, uid of the device
  metafile << "Device pid: " << devInfo->pid() << " vid: " << devInfo->vid()
           << " uid: " << devInfo->uid() << std::endl;

  // By getting the firmware version number of the device
  auto fwVer = devInfo->firmwareVersion();
  metafile << "Firmware version: " << fwVer << std::endl;

  // By getting the serial number of the device
  auto sn = devInfo->serialNumber();
  metafile << "Serial number: " << sn << std::endl;
}
// // Save raw data to file
void saveRawData(std::shared_ptr<ob::Frame> frame, int episode, int count) {
    // Create an output string stream
  std::ostringstream episode_pad;
  // Set the output width and fill character ('0' in our case)
  episode_pad << std::setw(5) << std::setfill('0') << episode;
  std::string episode_str = episode_pad.str();

  std::ostringstream count_pad;
  // Set the output width and fill character ('0' in our case)
  count_pad << std::setw(5) << std::setfill('0') << count;
  // Construct the filename using the string stream
  std::string count_str = count_pad.str();
  std::string fileName = "RawDepth_" + std::to_string(frame->timeStamp()) +
                         "ms_e" + episode_str + "_" +
                          count_str +
                          ".raw";
  std::ofstream outFile(fileName, std::ios::binary);
  if (!outFile.is_open()) {
    throw std::runtime_error("Failed to open file for writing: " + fileName);
  }
  outFile.write(reinterpret_cast<const char *>(frame->data()),
  frame->dataSize());

  outFile.close();
  std::cout << fileName << " saved successfully." << std::endl;
}

// Save the color image in PNG format
// (This function builds its own filename; you may modify it to include a folder
// path as needed.)
void saveColor(std::shared_ptr<ob::ColorFrame> colorFrame, int episode, int count) {
  if (!colorFrame) {
    std::cerr << "Color frame is null!" << std::endl;
    return;
  }

  cv::Mat colorMat;
  if (colorFrame->format() == OB_FORMAT_RGB888) {
    cv::Mat colorRawMat(colorFrame->height(), colorFrame->width(), CV_8UC3,
                        colorFrame->data());
    cv::cvtColor(colorRawMat, colorMat, cv::COLOR_RGB2BGR);
  } else if (colorFrame->format() == OB_FORMAT_MJPG) {
    ob::FormatConvertFilter formatConvertFilter;
    formatConvertFilter.setFormatConvertType(FORMAT_MJPG_TO_RGB888);
    auto convertedFrame =
        formatConvertFilter.process(colorFrame)->as<ob::ColorFrame>();
    cv::Mat colorRawMat(convertedFrame->height(), convertedFrame->width(),
                        CV_8UC3, convertedFrame->data());
    cv::cvtColor(colorRawMat, colorMat, cv::COLOR_RGB2BGR);
  } else if (colorFrame->format() == OB_FORMAT_YUYV) {
    cv::Mat colorRawMat(colorFrame->height(), colorFrame->width(), CV_8UC2,
                        colorFrame->data());
    cv::cvtColor(colorRawMat, colorMat, cv::COLOR_YUV2BGR_YUYV);
  } else {
    std::cerr << "Unsupported color format: " << colorFrame->format()
              << std::endl;
    return;
  }
  // Create an output string stream
  std::ostringstream episode_pad;
  // Set the output width and fill character ('0' in our case)
  episode_pad << std::setw(5) << std::setfill('0') << episode;
  std::string episode_str = episode_pad.str();

  std::ostringstream count_pad;
  // Set the output width and fill character ('0' in our case)
  count_pad << std::setw(5) << std::setfill('0') << count;
  // Construct the filename using the string stream
  std::string count_str = count_pad.str();
  std::vector<int> compression_params = {cv::IMWRITE_PNG_COMPRESSION, 0};
  std::string colorName = "Color_" + std::to_string(colorFrame->width()) + "x" +
                          std::to_string(colorFrame->height()) + "_" +
                          std::to_string(colorFrame->timeStamp()) + "ms_e" +
                          episode_str + "_" +
                          count_str +
                           ".png";
  // Save the color image
  cv::imwrite(colorName, colorMat, compression_params);
  std::cout << "Color saved: " << colorName << std::endl;
}

int main(int argc, char **argv) try {
  // Set logger severity via Context
  ob::Context::setLoggerSeverity(OB_LOG_SEVERITY_OFF);
  ob::Pipeline pipeline;

  // Configure streams
  std::shared_ptr<ob::Config> config = std::make_shared<ob::Config>();

  std::shared_ptr<ob::VideoStreamProfile> colorProfile = nullptr;
  try {
    auto colorProfiles = pipeline.getStreamProfileList(OB_SENSOR_COLOR);
    if (colorProfiles) {
      auto profile = colorProfiles->getProfile(OB_PROFILE_DEFAULT);
      colorProfile = profile->as<ob::VideoStreamProfile>();
    }
    config->enableStream(colorProfile);
  } catch (ob::Error &e) {
    config->setAlignMode(ALIGN_DISABLE);
    std::cerr << "Current device does not support color sensor!" << std::endl;
  }

  std::shared_ptr<ob::StreamProfileList> depthProfileList;
  OBAlignMode alignMode = ALIGN_DISABLE;
  if (colorProfile) {
    depthProfileList =
        pipeline.getD2CDepthProfileList(colorProfile, ALIGN_D2C_HW_MODE);
    if (depthProfileList->count() > 0)
      alignMode = ALIGN_D2C_HW_MODE;
    else {
      depthProfileList =
          pipeline.getD2CDepthProfileList(colorProfile, ALIGN_D2C_SW_MODE);
      if (depthProfileList->count() > 0) alignMode = ALIGN_D2C_SW_MODE;
    }
    pipeline.enableFrameSync();
  } else {
    depthProfileList = pipeline.getStreamProfileList(OB_SENSOR_DEPTH);
  }

  std::shared_ptr<ob::StreamProfile> depthProfile = nullptr;
  if (depthProfileList->count() > 0) {
    try {
      if (colorProfile) {
        depthProfile = depthProfileList->getVideoStreamProfile(
            OB_WIDTH_ANY, OB_HEIGHT_ANY, OB_FORMAT_ANY, colorProfile->fps());
      }
    } catch (...) {
      depthProfile = nullptr;
    }
    if (!depthProfile)
      depthProfile = depthProfileList->getProfile(OB_PROFILE_DEFAULT);
    config->enableStream(depthProfile);
  }
  config->setAlignMode(alignMode);

  // Start the pipeline
  pipeline.start(config);

  // Log metadata information to file and console
  std::ofstream metaFile("metadata.txt");
  if (!metaFile.is_open()) {
    std::cerr << "Could not open metadata file for writing." << std::endl;
  }

  // Log camera intrinsics used for point cloud projection.
  // The SDK uses the depth sensor intrinsics (depthIntrinsic) for generating
  // the point cloud.
  auto cameraParam = pipeline.getCameraParam();
  saveMetaDatatoFile(cameraParam, metaFile);
  // log alignment information
  if (alignMode == ALIGN_D2C_HW_MODE) {
    metaFile << "Alignment mode: D2C_HW_MODE\n";
  } else if (alignMode == ALIGN_D2C_SW_MODE) {
    metaFile << "Alignment mode: D2C_SW_MODE\n";
  } else {
    metaFile << "Alignment mode: DISABLE\n";
  }

  // Create a Context.
  ob::Context ctx;
  ctx.setDeviceChangedCallback([](std::shared_ptr<ob::DeviceList> removedList,
                                  std::shared_ptr<ob::DeviceList> addedList) {
    if (isWaitRebootComplete_) {
      if (addedList && addedList->deviceCount() > 0) {
        auto device = addedList->getDevice(0);
        if (isDeviceRemoved_ &&
            deviceSN_ == std::string(device->getDeviceInfo()->serialNumber())) {
          rebootedDevice_ = device;
          isWaitRebootComplete_ = false;

          std::unique_lock<std::mutex> lk(waitRebootMutex_);
          waitRebootCondition_.notify_all();
        }
      }

      if (removedList && removedList->deviceCount() > 0) {
        if (deviceUid_ == std::string(removedList->uid(0))) {
          isDeviceRemoved_ = true;
        }
      }
    }  // if isWaitRebootComplete_
  });

  // Query the list of connected devices
  auto devList = ctx.queryDeviceList();

  // Get the number of connected devices
  if (devList->deviceCount() == 0) {
    std::cerr << "Device not found!" << std::endl;
    return -1;
  }

  // Create a device, 0 means the index of the first device
  auto dev = devList->getDevice(0);
  dumpDeviceInfo(dev, metaFile);
  // Also log the RGB resolution if the color stream is enabled.
  if (colorProfile) {
    metaFile << "RGB Resolution: " << colorProfile->width() << " x "
             << colorProfile->height() << "\n";
    // log the fps from the colorprofile
    metaFile << "RGB FPS: " << colorProfile->fps() << "\n";
  }
  metaFile.close();

  // Set up point cloud filter using the retrieved camera parameters.
  // (By default the point cloud generator uses the depth intrinsics.)
  ob::PointCloudFilter pointCloud;
  pointCloud.setCameraParam(cameraParam);

  int count = 0;
  int episode = 0;
  while (true) {
    auto frameset = pipeline.waitForFrames(100);
    if (!frameset || !frameset->depthFrame() || !frameset->colorFrame()) {
      std::cout << "Frames not initially available, when warning stops, press 'r' to record, 'ESC' to exit.\n";
      continue;
    }
    if (kbhit()) {
      int key = getch();
      if (key == KEY_ESC) break;
      if (key == KEY_r || key == KEY_R) {
        std::cout << "Recording...\n";
        std::cout << "Press 's' to start episode.\n";
        int key = getch();
        if (key == KEY_s) {
        }
        while (true) {
          auto frameset = pipeline.waitForFrames(100);
          if (frameset && frameset->depthFrame() && frameset->colorFrame()) {
            saveRawData(frameset->depthFrame(), episode,count);
            saveColor(frameset->colorFrame(), episode,count);
            // prss p to stop episode
            int key = getch_noblock();
            if (key == KEY_p) {
              std::cout << "Episode stopped.\n";
              episode++;
              count = 0;
              break;
            }
            count++;
          } else {
            std::cout << "Frames not available.\n";
          }
        }
    }
  }
  }
    pipeline.stop();
    return 0;
  } catch (ob::Error &e) {
    std::cerr << "Error: " << e.getMessage() << std::endl;
    return EXIT_FAILURE;
  }
