
"use strict";

let TimeReference = require('./TimeReference.js');
let NavSatStatus = require('./NavSatStatus.js');
let MultiEchoLaserScan = require('./MultiEchoLaserScan.js');
let FluidPressure = require('./FluidPressure.js');
let Temperature = require('./Temperature.js');
let LaserScan = require('./LaserScan.js');
let JoyFeedbackArray = require('./JoyFeedbackArray.js');
let BatteryState = require('./BatteryState.js');
let MagneticField = require('./MagneticField.js');
let JoyFeedback = require('./JoyFeedback.js');
let ChannelFloat32 = require('./ChannelFloat32.js');
let PointCloud = require('./PointCloud.js');
let Illuminance = require('./Illuminance.js');
let MultiDOFJointState = require('./MultiDOFJointState.js');
let Range = require('./Range.js');
let CompressedImage = require('./CompressedImage.js');
let PointCloud2 = require('./PointCloud2.js');
let JointState = require('./JointState.js');
let Imu = require('./Imu.js');
let LaserEcho = require('./LaserEcho.js');
let NavSatFix = require('./NavSatFix.js');
let Image = require('./Image.js');
let RegionOfInterest = require('./RegionOfInterest.js');
let Joy = require('./Joy.js');
let PointField = require('./PointField.js');
let RelativeHumidity = require('./RelativeHumidity.js');
let CameraInfo = require('./CameraInfo.js');

module.exports = {
  TimeReference: TimeReference,
  NavSatStatus: NavSatStatus,
  MultiEchoLaserScan: MultiEchoLaserScan,
  FluidPressure: FluidPressure,
  Temperature: Temperature,
  LaserScan: LaserScan,
  JoyFeedbackArray: JoyFeedbackArray,
  BatteryState: BatteryState,
  MagneticField: MagneticField,
  JoyFeedback: JoyFeedback,
  ChannelFloat32: ChannelFloat32,
  PointCloud: PointCloud,
  Illuminance: Illuminance,
  MultiDOFJointState: MultiDOFJointState,
  Range: Range,
  CompressedImage: CompressedImage,
  PointCloud2: PointCloud2,
  JointState: JointState,
  Imu: Imu,
  LaserEcho: LaserEcho,
  NavSatFix: NavSatFix,
  Image: Image,
  RegionOfInterest: RegionOfInterest,
  Joy: Joy,
  PointField: PointField,
  RelativeHumidity: RelativeHumidity,
  CameraInfo: CameraInfo,
};
