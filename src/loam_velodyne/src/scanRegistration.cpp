// Copyright 2013, Ji Zhang, Carnegie Mellon University
// Further contributions copyright (c) 2016, Southwest Research Institute
// All rights reserved.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are met:
//
// 1. Redistributions of source code must retain the above copyright notice,
//    this list of conditions and the following disclaimer.
// 2. Redistributions in binary form must reproduce the above copyright notice,
//    this list of conditions and the following disclaimer in the documentation
//    and/or other materials provided with the distribution.
// 3. Neither the name of the copyright holder nor the names of its
//    contributors may be used to endorse or promote products derived from this
//    software without specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
// AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
// ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
// LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
// CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
// SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
// INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
// CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
// ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
// POSSIBILITY OF SUCH DAMAGE.
//
// This is an implementation of the algorithm described in the following paper:
//   J. Zhang and S. Singh. LOAM: Lidar Odometry and Mapping in Real-time.
//     Robotics: Science and Systems Conference (RSS). Berkeley, CA, July 2014.



/******************************Notes before reading [读前须知]*****************************/
/*imu is a right-handed coordinate system with x-axis forward, y-axis left, and z-axis upward.
Velodyne lidar is installed as a right-handed coordinate system with x-axis forward, y-axis left, and z-axis upward.
ScanRegistration will unify the two into a right-handed coordinate system with z-axis forward, x-axis left, and y-axis upward by exchanging the coordinate axes.
This is the coordinate system used in J. Zhang's paper.
After the exchange: R = Ry(yaw)*Rx(pitch)*Rz(roll)
[imu为x轴向前,y轴向左,z轴向上的右手坐标系，
  velodyne lidar被安装为x轴向前,y轴向左,z轴向上的右手坐标系，
  scanRegistration会把两者通过交换坐标轴，都统一到z轴向前,x轴向左,y轴向上的右手坐标系
  ，这是J. Zhang的论文里面使用的坐标系
  交换后：R = Ry(yaw)*Rx(pitch)*Rz(roll)]
*******************************************************************************/

#include <cmath>
#include <vector>

#include <loam_velodyne/common.h>
#include <opencv/cv.h>
#include <nav_msgs/Odometry.h>
#include <opencv/cv.h>
#include <pcl_conversions/pcl_conversions.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/kdtree/kdtree_flann.h>
#include <ros/ros.h>
#include <sensor_msgs/Imu.h>
#include <sensor_msgs/PointCloud2.h>
#include <tf/transform_datatypes.h>
#include <tf/transform_broadcaster.h>

using std::sin;
using std::cos;
using std::atan2;

//Scanning cycle, velodyne frequency 10Hz, period 0.1s [扫描周期, velodyne频率10Hz，周期0.1s}]
const double scanPeriod = 0.1;

//Initialize control variables [初始化控制变量]
const int systemDelay = 20;//Discard the first 20 frames of initial data [弃用前20帧初始数据]
int systemInitCount = 0;
bool systemInited = false;

//Number of laser radar lines [激光雷达线数]
const int N_SCANS = 16;

//Point cloud curvature, 40000 is the maximum number of points in a frame of point cloud [点云曲率, 40000为一帧点云中点的最大数量]
float cloudCurvature[40000];
//Serial number corresponding to the curvature point [曲率点对应的序号]
int cloudSortInd[40000];
//Point filter flag: 0-not filtered, 1-filtered [点是否筛选过标志：0-未筛选过，1-筛选过]
int cloudNeighborPicked[40000];
//Point classification number: 2-represents large curvature, 1-represents relatively large curvature, -1-represents small curvature, 0-relatively small curvature 
//(where 1 includes 2, 0 includes 1, 0 and 1 constitute all the points in the point cloud)
//[点分类标号:2-代表曲率很大，1-代表曲率比较大,-1-代表曲率很小，0-曲率比较小(其中1包含了2,0包含了1,0和1构成了点云全部的点)]
int cloudLabel[40000];

//The position where the imu timestamp is greater than the current point cloud timestamp [imu时间戳大于当前点云时间戳的位置]
int imuPointerFront = 0;
//The position of the most recently received point in the imu array [imu最新收到的点在数组中的位置]
int imuPointerLast = -1;
//The length of the imu circular(?) queue [imu循环队列长度]
const int imuQueLength = 200;

//Displacement/velocity/Euler angle of the first point of the point cloud data [点云数据开始第一个点的位移/速度/欧拉角]
float imuRollStart = 0, imuPitchStart = 0, imuYawStart = 0;
float imuRollCur = 0, imuPitchCur = 0, imuYawCur = 0;

float imuVeloXStart = 0, imuVeloYStart = 0, imuVeloZStart = 0;
float imuShiftXStart = 0, imuShiftYStart = 0, imuShiftZStart = 0;

//Speed ​​and displacement information of the current point [当前点的速度，位移信息]
float imuVeloXCur = 0, imuVeloYCur = 0, imuVeloZCur = 0;
float imuShiftXCur = 0, imuShiftYCur = 0, imuShiftZCur = 0;

//Distortion displacement and speed of the current point in each point cloud data relative to the first point [每次点云数据当前点相对于开始第一个点的畸变位移，速度]
float imuShiftFromStartXCur = 0, imuShiftFromStartYCur = 0, imuShiftFromStartZCur = 0;
float imuVeloFromStartXCur = 0, imuVeloFromStartYCur = 0, imuVeloFromStartZCur = 0;

//IMU information [IMU信息]
double imuTime[imuQueLength] = {0};
float imuRoll[imuQueLength] = {0};
float imuPitch[imuQueLength] = {0};
float imuYaw[imuQueLength] = {0};

float imuAccX[imuQueLength] = {0};
float imuAccY[imuQueLength] = {0};
float imuAccZ[imuQueLength] = {0};

float imuVeloX[imuQueLength] = {0};
float imuVeloY[imuQueLength] = {0};
float imuVeloZ[imuQueLength] = {0};

float imuShiftX[imuQueLength] = {0};
float imuShiftY[imuQueLength] = {0};
float imuShiftZ[imuQueLength] = {0};

ros::Publisher pubLaserCloud;
ros::Publisher pubCornerPointsSharp;
ros::Publisher pubCornerPointsLessSharp;
ros::Publisher pubSurfPointsFlat;
ros::Publisher pubSurfPointsLessFlat;
ros::Publisher pubImuTrans;

//Calculate the displacement distortion caused by acceleration and deceleration motion of the points in the point cloud relative to the first starting point in the local coordinate system
//[计算局部坐标系下点云中的点相对第一个开始点的由于加减速运动产生的位移畸变]
void ShiftToStartIMU(float pointTime)
{
  //Calculate the distortion displacement caused by acceleration and deceleration relative to the first point (distortion displacement delta_Tg in the global coordinate system)
  //[计算相对于第一个点由于加减速产生的畸变位移(全局坐标系下畸变位移量delta_Tg)]
  //imuShiftFromStartCur = imuShiftCur - (imuShiftStart + imuVeloStart * pointTime)
  imuShiftFromStartXCur = imuShiftXCur - imuShiftXStart - imuVeloXStart * pointTime;
  imuShiftFromStartYCur = imuShiftYCur - imuShiftYStart - imuVeloYStart * pointTime;
  imuShiftFromStartZCur = imuShiftZCur - imuShiftZStart - imuVeloZStart * pointTime;

  /********************************************************************************
  Rz(pitch).inverse * Rx(pitch).inverse * Ry(yaw).inverse * delta_Tg
  transfrom from the global frame to the local frame
  *********************************************************************************/

  //Rotate around the y-axis (-imuYawStart), i.e. Ry(yaw).inverse [绕y轴旋转(-imuYawStart)，即Ry(yaw).inverse]
  float x1 = cos(imuYawStart) * imuShiftFromStartXCur - sin(imuYawStart) * imuShiftFromStartZCur;
  float y1 = imuShiftFromStartYCur;
  float z1 = sin(imuYawStart) * imuShiftFromStartXCur + cos(imuYawStart) * imuShiftFromStartZCur;

  //Rotate around the x-axis (-imuPitchStart), i.e. Rx(pitch).inverse [绕x轴旋转(-imuPitchStart)，即Rx(pitch).inverse]
  float x2 = x1;
  float y2 = cos(imuPitchStart) * y1 + sin(imuPitchStart) * z1;
  float z2 = -sin(imuPitchStart) * y1 + cos(imuPitchStart) * z1;

  //Rotate around the z-axis (-imuRollStart), i.e. Rz(pitch).inverse [绕z轴旋转(-imuRollStart)，即Rz(pitch).inverse]
  imuShiftFromStartXCur = cos(imuRollStart) * x2 + sin(imuRollStart) * y2;
  imuShiftFromStartYCur = -sin(imuRollStart) * x2 + cos(imuRollStart) * y2;
  imuShiftFromStartZCur = z2;
}

//Calculate the velocity distortion (increment) caused by acceleration and deceleration of the points in the point cloud in the local coordinate system relative to the first starting point
//[计算局部坐标系下点云中的点相对第一个开始点由于加减速产生的的速度畸变（增量]
void VeloToStartIMU()
{
  //Calculate the distortion velocity due to acceleration and deceleration relative to the first point (distortion velocity increment delta_Vg in the global coordinate system)
  //[计算相对于第一个点由于加减速产生的畸变速度(全局坐标系下畸变速度增量delta_Vg)]
  imuVeloFromStartXCur = imuVeloXCur - imuVeloXStart;
  imuVeloFromStartYCur = imuVeloYCur - imuVeloYStart;
  imuVeloFromStartZCur = imuVeloZCur - imuVeloZStart;

  /********************************************************************************
    Rz(pitch).inverse * Rx(pitch).inverse * Ry(yaw).inverse * delta_Vg
    transfrom from the global frame to the local frame
  *********************************************************************************/

  //Rotate around the y axis (-imuYawStart), i.e. Ry(yaw).inverse [绕y轴旋转(-imuYawStart)，即Ry(yaw).inverse]
  float x1 = cos(imuYawStart) * imuVeloFromStartXCur - sin(imuYawStart) * imuVeloFromStartZCur;
  float y1 = imuVeloFromStartYCur;
  float z1 = sin(imuYawStart) * imuVeloFromStartXCur + cos(imuYawStart) * imuVeloFromStartZCur;

  //Rotate around the x-axis (-imuPitchStart), i.e. Rx(pitch).inverse [绕x轴旋转(-imuPitchStart)，即Rx(pitch).inverse]
  float x2 = x1;
  float y2 = cos(imuPitchStart) * y1 + sin(imuPitchStart) * z1;
  float z2 = -sin(imuPitchStart) * y1 + cos(imuPitchStart) * z1;

  //Rotate around the z-axis (-imuRollStart), i.e. Rz(pitch).inverse [绕z轴旋转(-imuRollStart)，即Rz(pitch).inverse]
  imuVeloFromStartXCur = cos(imuRollStart) * x2 + sin(imuRollStart) * y2;
  imuVeloFromStartYCur = -sin(imuRollStart) * x2 + cos(imuRollStart) * y2;
  imuVeloFromStartZCur = z2;
}

//Remove displacement distortion from point cloud acceleration and deceleration [去除点云加减速产生的位移畸变]
void TransformToStartIMU(PointType *p)
{
  /********************************************************************************
    Ry*Rx*Rz*Pl, transform point to the global frame
  *********************************************************************************/
  //Rotate around the z-axis (imuRollCur) [绕z轴旋转(imuRollCur)]
  float x1 = cos(imuRollCur) * p->x - sin(imuRollCur) * p->y;
  float y1 = sin(imuRollCur) * p->x + cos(imuRollCur) * p->y;
  float z1 = p->z;

  //Rotate around the x-axis (imuPitchCur) [绕x轴旋转(imuPitchCur)]
  float x2 = x1;
  float y2 = cos(imuPitchCur) * y1 - sin(imuPitchCur) * z1;
  float z2 = sin(imuPitchCur) * y1 + cos(imuPitchCur) * z1;

  //Rotate around the y-axis (imuYawCur) [绕y轴旋转(imuYawCur)]
  float x3 = cos(imuYawCur) * x2 + sin(imuYawCur) * z2;
  float y3 = y2;
  float z3 = -sin(imuYawCur) * x2 + cos(imuYawCur) * z2;

  /********************************************************************************
    Rz(pitch).inverse * Rx(pitch).inverse * Ry(yaw).inverse * Pg
    transfrom global points to the local frame
  *********************************************************************************/
  
  //Rotation around y-axis (-imuYawStart) [绕y轴旋转(-imuYawStart)]
  float x4 = cos(imuYawStart) * x3 - sin(imuYawStart) * z3;
  float y4 = y3;
  float z4 = sin(imuYawStart) * x3 + cos(imuYawStart) * z3;

  //Rotate around x-axis (-imuPitchStart) [绕x轴旋转(-imuPitchStart)]
  float x5 = x4;
  float y5 = cos(imuPitchStart) * y4 + sin(imuPitchStart) * z4;
  float z5 = -sin(imuPitchStart) * y4 + cos(imuPitchStart) * z4;

  //Rotate(-imuRollStart) around the z-axis and then superimpose the translations [绕z轴旋转(-imuRollStart)，然后叠加平移量]
  p->x = cos(imuRollStart) * x5 + sin(imuRollStart) * y5 + imuShiftFromStartXCur;
  p->y = -sin(imuRollStart) * x5 + cos(imuRollStart) * y5 + imuShiftFromStartYCur;
  p->z = z5 + imuShiftFromStartZCur;
}

//Integrate velocity and displacement [积分速度与位移]
void AccumulateIMUShift()
{
  float roll = imuRoll[imuPointerLast];
  float pitch = imuPitch[imuPointerLast];
  float yaw = imuYaw[imuPointerLast];
  float accX = imuAccX[imuPointerLast];
  float accY = imuAccY[imuPointerLast];
  float accZ = imuAccZ[imuPointerLast];

  //The acceleration value at the current moment is rotated (roll, pitch, yaw) around the exchanged ZXY fixed axis (the original XYZ) 
  //to get the acceleration value in the world coordinate system (right hand rule).
  //[将当前时刻的加速度值绕交换过的ZXY固定轴（原XYZ）分别旋转(roll, pitch, yaw)角，转换得到世界坐标系下的加速度值(right hand rule)]
  //Roll around the z-axis. [绕z轴旋转(roll)]
  float x1 = cos(roll) * accX - sin(roll) * accY;
  float y1 = sin(roll) * accX + cos(roll) * accY;
  float z1 = accZ;
  //Rotate around the x-axis (pitch). [绕x轴旋转(pitch)]
  float x2 = x1;
  float y2 = cos(pitch) * y1 - sin(pitch) * z1;
  float z2 = sin(pitch) * y1 + cos(pitch) * z1;
  //Rotate around the y-axis (yaw). [绕y轴旋转(yaw)]
  accX = cos(yaw) * x2 + sin(yaw) * z2;
  accY = y2;
  accZ = -sin(yaw) * x2 + cos(yaw) * z2;

  //Previous imu point [上一个imu点]
  int imuPointerBack = (imuPointerLast + imuQueLength - 1) % imuQueLength;
  //Time elapsed from the last point to the current point, i.e., the period of the imu measurement is calculated. [上一个点到当前点所经历的时间，即计算imu测量周期]
  double timeDiff = imuTime[imuPointerLast] - imuTime[imuPointerBack];
  //The frequency of the imu should be at least higher than the lidar, so that the imu information can be used, and the later correction is meaningful.
  //[要求imu的频率至少比lidar高，这样的imu信息才使用，后面校正也才有意义]
  if (timeDiff < scanPeriod) {// (implied motion from rest) [（隐含从静止开始运动)]
    //Find the displacement and velocity at each imu time point, between which the motion is considered to be uniformly accelerated linear motion. [求每个imu时间点的位移与速度,两点之间视为匀加速直线运动]
    imuShiftX[imuPointerLast] = imuShiftX[imuPointerBack] + imuVeloX[imuPointerBack] * timeDiff 
                              + accX * timeDiff * timeDiff / 2;
    imuShiftY[imuPointerLast] = imuShiftY[imuPointerBack] + imuVeloY[imuPointerBack] * timeDiff 
                              + accY * timeDiff * timeDiff / 2;
    imuShiftZ[imuPointerLast] = imuShiftZ[imuPointerBack] + imuVeloZ[imuPointerBack] * timeDiff 
                              + accZ * timeDiff * timeDiff / 2;

    imuVeloX[imuPointerLast] = imuVeloX[imuPointerBack] + accX * timeDiff;
    imuVeloY[imuPointerLast] = imuVeloY[imuPointerBack] + accY * timeDiff;
    imuVeloZ[imuPointerLast] = imuVeloZ[imuPointerBack] + accZ * timeDiff;
  }
}

//Receive point cloud data, velodyne radar coordinate system is installed as x-axis forward, y-axis to the left, z-axis up right-handed coordinate system.
//[接收点云数据，velodyne雷达坐标系安装为x轴向前，y轴向左，z轴向上的右手坐标系]
void laserCloudHandler(const sensor_msgs::PointCloud2ConstPtr& laserCloudMsg)
{
  if (!systemInited) {//Discard the first 20 point cloud data [丢弃前20个点云数据]
    systemInitCount++;
    if (systemInitCount >= systemDelay) {
      systemInited = true;
    }
    return;
  }

  //Record the start and end indices of each point with curvature in each scan. [记录每个scan有曲率的点的开始和结束索引]
  std::vector<int> scanStartInd(N_SCANS, 0);
  std::vector<int> scanEndInd(N_SCANS, 0);
  
  //Current point cloud time [当前点云时间]
  double timeScanCur = laserCloudMsg->header.stamp.toSec();
  pcl::PointCloud<pcl::PointXYZ> laserCloudIn;
  //Convert messages to pcl data for storage [消息转换成pcl数据存放]
  pcl::fromROSMsg(*laserCloudMsg, laserCloudIn);
  std::vector<int> indices;
  //Remove empty points [移除空点]
  pcl::removeNaNFromPointCloud(laserCloudIn, laserCloudIn, indices);
  //Number of point cloud points [点云点的数量]
  int cloudSize = laserCloudIn.points.size();
  //lidar rotation angle of the start point of the scan, atan2 range [-pi,+pi], the rotation angle is calculated with a negative sign because velodyne rotates clockwise.
  //[lidar scan开始点的旋转角,atan2范围[-pi,+pi],计算旋转角时取负号是因为velodyne是顺时针旋转]
  float startOri = -atan2(laserCloudIn.points[0].y, laserCloudIn.points[0].x);
  //lidar scan the end of the rotation angle of the point, plus 2 * pi so that the point cloud rotation period of 2 * pi [lidar scan结束点的旋转角，加2*pi使点云旋转周期为2*pi]
  float endOri = -atan2(laserCloudIn.points[cloudSize - 1].y,
                        laserCloudIn.points[cloudSize - 1].x) + 2 * M_PI;

  //The difference between the end azimuth and the start azimuth is controlled to be within (PI,3*PI), allowing the lidar not to be a circular scan.
  //[结束方位角与开始方位角差值控制在(PI,3*PI)范围，允许lidar不是一个圆周扫描]
  //Normally within this range: pi < endOri - startOri < 3*pi, anomalies are corrected [正常情况下在这个范围内：pi < endOri - startOri < 3*pi，异常则修正]
  if (endOri - startOri > 3 * M_PI) {
    endOri -= 2 * M_PI;
  } else if (endOri - startOri < M_PI) {
    endOri += 2 * M_PI;
  }
  //Is the lidar scan line rotated more than half? [lidar扫描线是否旋转过半]
  bool halfPassed = false;
  int count = cloudSize;
  PointType point;
  std::vector<pcl::PointCloud<PointType> > laserCloudScans(N_SCANS);
  for (int i = 0; i < cloudSize; i++) {
    //The coordinate axes are swapped, and the velodyne lidar coordinate system is also converted to a right-handed coordinate system with the z axis facing forward and the x axis facing left.
    //[坐标轴交换，velodyne lidar的坐标系也转换到z轴向前，x轴向左的右手坐标系]
    point.x = laserCloudIn.points[i].y;
    point.y = laserCloudIn.points[i].z;
    point.z = laserCloudIn.points[i].x;

    //Calculate the elevation angle of the point (according to the vertical angle calculation formula in the lidar document), arrange the laser line number according to the elevation angle, 
    //and the interval between each two scans of the velodyne is 2 degrees
    //[计算点的仰角(根据lidar文档垂直角计算公式),根据仰角排列激光线号，velodyne每两个scan之间间隔2度]
    float angle = atan(point.y / sqrt(point.x * point.x + point.z * point.z)) * 180 / M_PI;
    int scanID;
    //Elevation angle rounding (plus or minus 0.5 truncation effect equals rounding) [仰角四舍五入(加减0.5截断效果等于四舍五入)]
    int roundedAngle = int(angle + (angle<0.0?-0.5:+0.5)); 
    if (roundedAngle > 0){
      scanID = roundedAngle;
    }
    else {
      scanID = roundedAngle + (N_SCANS - 1);
    }
    //Filter points, only pick points in the range of [-15 degrees, +15 degrees], scanID belongs to [0,15] {过滤点，只挑选[-15度，+15度]范围内的点,scanID属于[0,15]}
    if (scanID > (N_SCANS - 1) || scanID < 0 ){
      count--;
      continue;
    }

    //Angle of rotation of the point [该点的旋转角]
    float ori = -atan2(point.x, point.z);
    //Compensation is performed by selecting the difference between the start position and the end position depending on whether the scan line is rotated by half.
    //[根据扫描线是否旋转过半选择与起始位置还是终止位置进行差值计算，从而进行补偿]
    if (!halfPassed) {
        //Make sure that -pi/2 < ori - startOri < 3*pi/2 [确保-pi/2 < ori - startOri < 3*pi/2]
      if (ori < startOri - M_PI / 2) {
        ori += 2 * M_PI;
      } else if (ori > startOri + M_PI * 3 / 2) {
        ori -= 2 * M_PI;
      }

      if (ori - startOri > M_PI) {
        halfPassed = true;
      }
    } else {
      ori += 2 * M_PI;

      //Make sure -3*pi/2 < ori - endOri < pi/2 [确保-3*pi/2 < ori - endOri < pi/2]
      if (ori < endOri - M_PI * 3 / 2) {
        ori += 2 * M_PI;
      } else if (ori > endOri + M_PI / 2) {
        ori -= 2 * M_PI;
      } 
    }

    //-0.5 < relTime < 1.5 (the ratio of the angle of rotation of the point to the angle of rotation of the whole cycle, i.e. the relative time of the point in the point cloud)
    //[（点旋转的角度与整个周期旋转角度的比率, 即点云中点的相对时间]
    float relTime = (ori - startOri) / (endOri - startOri);
    //Point intensity = line number + point relative time (i.e., an integer + a decimal, the integer part is the line number, the decimal part is the relative time of the point),Uniform scanning: 
    //calculate the time relative to the scanning start position based on the current scanning angle and scanning period.
    //[点强度=线号+点相对时间（即一个整数+一个小数，整数部分是线号，小数部分是该点的相对时间）,匀速扫描：根据当前扫描的角度和扫描周期计算相对扫描起始位置的时间]
    point.intensity = scanID + scanPeriod * relTime;

    //Point time = point cloud time + cycle time [点时间=点云时间+周期时间]
    if (imuPointerLast >= 0) {//If IMU data is received, use IMU to correct point cloud distortion. [如果收到IMU数据,使用IMU矫正点云畸变]
      float pointTime = relTime * scanPeriod;//Calculate the cycle time of the point [计算点的周期时间]
      //Find if the point cloud timestamp is less than the IMU timestamp at the IMU position: imuPointerFront [寻找是否有点云的时间戳小于IMU的时间戳的IMU位置:imuPointerFront]
      while (imuPointerFront != imuPointerLast) {
        if (timeScanCur + pointTime < imuTime[imuPointerFront]) {
          break;
        }
        imuPointerFront = (imuPointerFront + 1) % imuQueLength;
      }

      //Not found, then imuPointerFront==imtPointerLast, can only use the velocity, displacement and Euler angle of the latest IMU received as the velocity, displacement 
      //and Euler angle of the current point.
      //[没找到,此时imuPointerFront==imtPointerLast,只能以当前收到的最新的IMU的速度，位移，欧拉角作为当前点的速度，位移，欧拉角使用]
      if (timeScanCur + pointTime > imuTime[imuPointerFront]) {
        imuRollCur = imuRoll[imuPointerFront];
        imuPitchCur = imuPitch[imuPointerFront];
        imuYawCur = imuYaw[imuPointerFront];

        imuVeloXCur = imuVeloX[imuPointerFront];
        imuVeloYCur = imuVeloY[imuPointerFront];
        imuVeloZCur = imuVeloZ[imuPointerFront];

        imuShiftXCur = imuShiftX[imuPointerFront];
        imuShiftYCur = imuShiftY[imuPointerFront];
        imuShiftZCur = imuShiftZ[imuPointerFront];
        //Find the IMU position where the point cloud timestamp is less than the IMU timestamp, then the point must be between imuPointerBack and imuPointerFront, according to this linear interpolation, 
        //calculate the velocity, displacement and Euler angle of the point cloud point.
        //[找到了点云时间戳小于IMU时间戳的IMU位置,则该点必处于imuPointerBack和imuPointerFront之间，据此线性插值，计算点云点的速度，位移和欧拉角]
      } else {
        int imuPointerBack = (imuPointerFront + imuQueLength - 1) % imuQueLength;
        //Calculate the weight distribution ratio by time distance, i.e. linear interpolation. [按时间距离计算权重分配比率,也即线性插值]
        float ratioFront = (timeScanCur + pointTime - imuTime[imuPointerBack]) 
                         / (imuTime[imuPointerFront] - imuTime[imuPointerBack]);
        float ratioBack = (imuTime[imuPointerFront] - timeScanCur - pointTime) 
                        / (imuTime[imuPointerFront] - imuTime[imuPointerBack]);

        imuRollCur = imuRoll[imuPointerFront] * ratioFront + imuRoll[imuPointerBack] * ratioBack;
        imuPitchCur = imuPitch[imuPointerFront] * ratioFront + imuPitch[imuPointerBack] * ratioBack;
        if (imuYaw[imuPointerFront] - imuYaw[imuPointerBack] > M_PI) {
          imuYawCur = imuYaw[imuPointerFront] * ratioFront + (imuYaw[imuPointerBack] + 2 * M_PI) * ratioBack;
        } else if (imuYaw[imuPointerFront] - imuYaw[imuPointerBack] < -M_PI) {
          imuYawCur = imuYaw[imuPointerFront] * ratioFront + (imuYaw[imuPointerBack] - 2 * M_PI) * ratioBack;
        } else {
          imuYawCur = imuYaw[imuPointerFront] * ratioFront + imuYaw[imuPointerBack] * ratioBack;
        }

        //Essence(?) [本质]:imuVeloXCur = imuVeloX[imuPointerback] + (imuVelX[imuPointerFront]-imuVelX[imuPoniterBack])*ratioFront
        imuVeloXCur = imuVeloX[imuPointerFront] * ratioFront + imuVeloX[imuPointerBack] * ratioBack;
        imuVeloYCur = imuVeloY[imuPointerFront] * ratioFront + imuVeloY[imuPointerBack] * ratioBack;
        imuVeloZCur = imuVeloZ[imuPointerFront] * ratioFront + imuVeloZ[imuPointerBack] * ratioBack;

        imuShiftXCur = imuShiftX[imuPointerFront] * ratioFront + imuShiftX[imuPointerBack] * ratioBack;
        imuShiftYCur = imuShiftY[imuPointerFront] * ratioFront + imuShiftY[imuPointerBack] * ratioBack;
        imuShiftZCur = imuShiftZ[imuPointerFront] * ratioFront + imuShiftZ[imuPointerBack] * ratioBack;
      }

      if (i == 0) {//If it's the first point, memorize the velocity, displacement, and Euler angle at the start of the point cloud. [如果是第一个点,记住点云起始位置的速度，位移，欧拉角]
        imuRollStart = imuRollCur;
        imuPitchStart = imuPitchCur;
        imuYawStart = imuYawCur;

        imuVeloXStart = imuVeloXCur;
        imuVeloYStart = imuVeloYCur;
        imuVeloZStart = imuVeloZCur;

        imuShiftXStart = imuShiftXCur;
        imuShiftYStart = imuShiftYCur;
        imuShiftZStart = imuShiftZCur;
        //Calculate the displacement and velocity distortion of each point relative to the first point due to acceleration and deceleration and non-uniform motion, 
        //and re-compensate and correct the position information of each point in the point cloud.
        //[计算之后每个点相对于第一个点的由于加减速非匀速运动产生的位移速度畸变，并对点云中的每个点位置信息重新补偿矫正]
      } else {
        ShiftToStartIMU(pointTime);
        VeloToStartIMU();
        TransformToStartIMU(&point);
      }
    }
    laserCloudScans[scanID].push_back(point);//Place each compensated point into a container with the corresponding line number. [将每个补偿矫正的点放入对应线号的容器]
  }

  //Obtain the number of points within the valid range [获得有效范围内的点的数量]
  cloudSize = count;

  pcl::PointCloud<PointType>::Ptr laserCloud(new pcl::PointCloud<PointType>());
  for (int i = 0; i < N_SCANS; i++) {//Put all the points into a container according to the line number from smallest to largest. [将所有的点按照线号从小到大放入一个容器]
    *laserCloud += laserCloudScans[i];
  }
  int scanCount = -1;
  ////Calculate the curvature using the five points before and after each point, so the first and last five points are skipped. [使用每个点的前后五个点计算曲率，因此前五个与最后五个点跳过]
  for (int i = 5; i < cloudSize - 5; i++) {
    float diffX = laserCloud->points[i - 5].x + laserCloud->points[i - 4].x 
                + laserCloud->points[i - 3].x + laserCloud->points[i - 2].x 
                + laserCloud->points[i - 1].x - 10 * laserCloud->points[i].x 
                + laserCloud->points[i + 1].x + laserCloud->points[i + 2].x
                + laserCloud->points[i + 3].x + laserCloud->points[i + 4].x
                + laserCloud->points[i + 5].x;
    float diffY = laserCloud->points[i - 5].y + laserCloud->points[i - 4].y 
                + laserCloud->points[i - 3].y + laserCloud->points[i - 2].y 
                + laserCloud->points[i - 1].y - 10 * laserCloud->points[i].y 
                + laserCloud->points[i + 1].y + laserCloud->points[i + 2].y
                + laserCloud->points[i + 3].y + laserCloud->points[i + 4].y
                + laserCloud->points[i + 5].y;
    float diffZ = laserCloud->points[i - 5].z + laserCloud->points[i - 4].z 
                + laserCloud->points[i - 3].z + laserCloud->points[i - 2].z 
                + laserCloud->points[i - 1].z - 10 * laserCloud->points[i].z 
                + laserCloud->points[i + 1].z + laserCloud->points[i + 2].z
                + laserCloud->points[i + 3].z + laserCloud->points[i + 4].z
                + laserCloud->points[i + 5].z;
    //Curvature calculation [曲率计算]
    cloudCurvature[i] = diffX * diffX + diffY * diffY + diffZ * diffZ;
    //Record the index of the curvature point [记录曲率点的索引]
    cloudSortInd[i] = i;
    //Initially, the points are all unfiltered [初始时，点全未筛选过]
    cloudNeighborPicked[i] = 0;
    //Initialize to less flat points [初始化为less flat点]
    cloudLabel[i] = 0;

    //For each scan, only the first point that matches will come in, because the points for each scan are stored together. [每个scan，只有第一个符合的点会进来，因为每个scan的点都在一起存放]
    if (int(laserCloud->points[i].intensity) != scanCount) {
      scanCount = int(laserCloud->points[i].intensity);//Controlling only the first point in each scan [控制每个scan只进入第一个点]

      //The curvature is only taken from the same scan, the curvature calculated across scans is illegal and excluded, i.e. the five points before and after each scan are excluded.
      //[曲率只取同一个scan计算出来的，跨scan计算的曲率非法，排除，也即排除每个scan的前后五个点]
      if (scanCount > 0 && scanCount < N_SCANS) {
        scanStartInd[scanCount] = i + 5;
        scanEndInd[scanCount - 1] = i - 5;
      }
    }
  }
  //The first scan curvature point valid point sequence starts from the 5th, and the last laser line end point sequence size-5 [第一个scan曲率点有效点序从第5个开始，最后一个激光线结束点序size-5]
  scanStartInd[0] = 5;
  scanEndInd.back() = cloudSize - 5;

  //Pick points, exclude points that are easily blocked by the ramp and outliers, some points are easily blocked by the ramp and outliers may appear by chance, 
  //these situations may cause the two scans to not be seen at the same time.
  //[挑选点，排除容易被斜面挡住的点以及离群点，有些点容易被斜面挡住，而离群点可能出现带有偶然性，这些情况都可能导致前后两次扫描不能被同时看到]
  for (int i = 5; i < cloudSize - 6; i++) {//Difference with the latter point, so minus 6 [与后一个点差值，所以减6]
    float diffX = laserCloud->points[i + 1].x - laserCloud->points[i].x;
    float diffY = laserCloud->points[i + 1].y - laserCloud->points[i].y;
    float diffZ = laserCloud->points[i + 1].z - laserCloud->points[i].z;
    //Calculate the sum of the squares of the distances between the effective curvature point and the latter point [计算有效曲率点与后一个点之间的距离平方和]
    float diff = diffX * diffX + diffY * diffY + diffZ * diffZ;

    if (diff > 0.1) {//Prerequisite: the distance between two points should be greater than 0.1 [前提:两个点之间距离要大于0.1]

      //Depth of point [点的深度]
      float depth1 = sqrt(laserCloud->points[i].x * laserCloud->points[i].x + 
                     laserCloud->points[i].y * laserCloud->points[i].y +
                     laserCloud->points[i].z * laserCloud->points[i].z);

      //Depth of the latter point [后一个点的深度]
      float depth2 = sqrt(laserCloud->points[i + 1].x * laserCloud->points[i + 1].x + 
                     laserCloud->points[i + 1].y * laserCloud->points[i + 1].y +
                     laserCloud->points[i + 1].z * laserCloud->points[i + 1].z);

      //Calculate the distance by pulling back the point with the greater depth in proportion to the depth of the two points [按照两点的深度的比例，将深度较大的点拉回后计算距离]
      if (depth1 > depth2) {
        diffX = laserCloud->points[i + 1].x - laserCloud->points[i].x * depth2 / depth1;
        diffY = laserCloud->points[i + 1].y - laserCloud->points[i].y * depth2 / depth1;
        diffZ = laserCloud->points[i + 1].z - laserCloud->points[i].z * depth2 / depth1;

        //Side length ratio is also the radian value, if it is less than 0.1, it means that the angle is smaller, the slope is steeper, the depth of the point changes more drastically, 
        //the point is in the approximate parallel to the laser beam on the slope. [边长比也即是弧度值，若小于0.1，说明夹角比较小，斜面比较陡峭,点深度变化比较剧烈,点处在近似与激光束平行的斜面上]
        if (sqrt(diffX * diffX + diffY * diffY + diffZ * diffZ) / depth2 < 0.1) {//Eliminate points that are easily blocked by the inclined plane [排除容易被斜面挡住的点]
            //This point and the five previous points (mostly on the slope) are all set as filtered [该点及前面五个点（大致都在斜面上）全部置为筛选过]
          cloudNeighborPicked[i - 5] = 1;
          cloudNeighborPicked[i - 4] = 1;
          cloudNeighborPicked[i - 3] = 1;
          cloudNeighborPicked[i - 2] = 1;
          cloudNeighborPicked[i - 1] = 1;
          cloudNeighborPicked[i] = 1;
        }
      } else {
        diffX = laserCloud->points[i + 1].x * depth1 / depth2 - laserCloud->points[i].x;
        diffY = laserCloud->points[i + 1].y * depth1 / depth2 - laserCloud->points[i].y;
        diffZ = laserCloud->points[i + 1].z * depth1 / depth2 - laserCloud->points[i].z;

        if (sqrt(diffX * diffX + diffY * diffY + diffZ * diffZ) / depth1 < 0.1) {
          cloudNeighborPicked[i + 1] = 1;
          cloudNeighborPicked[i + 2] = 1;
          cloudNeighborPicked[i + 3] = 1;
          cloudNeighborPicked[i + 4] = 1;
          cloudNeighborPicked[i + 5] = 1;
          cloudNeighborPicked[i + 6] = 1;
        }
      }
    }

    float diffX2 = laserCloud->points[i].x - laserCloud->points[i - 1].x;
    float diffY2 = laserCloud->points[i].y - laserCloud->points[i - 1].y;
    float diffZ2 = laserCloud->points[i].z - laserCloud->points[i - 1].z;
    //The sum of the squares of the distances to the previous point [与前一个点的距离平方和]
    float diff2 = diffX2 * diffX2 + diffY2 * diffY2 + diffZ2 * diffZ2;

    //The sum of the squares of the depths of the points [点深度的平方和]
    float dis = laserCloud->points[i].x * laserCloud->points[i].x
              + laserCloud->points[i].y * laserCloud->points[i].y
              + laserCloud->points[i].z * laserCloud->points[i].z;

    //Points where the sum of the squares of the preceding and following points is greater than two ten thousandths of the sum of the squares of the depths are considered to be outliers, 
    //including points on steeply sloping surfaces, strongly convex and concave points, and some points in open areas, and are filtered and discarded.
    //[与前后点的平方和都大于深度平方和的万分之二，这些点视为离群点，包括陡斜面上的点，强烈凸凹点和空旷区域中的某些点，置为筛选过，弃用]
    if (diff > 0.0002 * dis && diff2 > 0.0002 * dis) {
      cloudNeighborPicked[i] = 1;
    }
  }


  pcl::PointCloud<PointType> cornerPointsSharp;
  pcl::PointCloud<PointType> cornerPointsLessSharp;
  pcl::PointCloud<PointType> surfPointsFlat;
  pcl::PointCloud<PointType> surfPointsLessFlat;

  //将每条线上的点分入相应的类别：边沿点和平面点
  for (int i = 0; i < N_SCANS; i++) {
    pcl::PointCloud<PointType>::Ptr surfPointsLessFlatScan(new pcl::PointCloud<PointType>);
    //将每个scan的曲率点分成6等份处理,确保周围都有点被选作特征点
    for (int j = 0; j < 6; j++) {
        //六等份起点：sp = scanStartInd + (scanEndInd - scanStartInd)*j/6
      int sp = (scanStartInd[i] * (6 - j)  + scanEndInd[i] * j) / 6;
      //六等份终点：ep = scanStartInd - 1 + (scanEndInd - scanStartInd)*(j+1)/6
      int ep = (scanStartInd[i] * (5 - j)  + scanEndInd[i] * (j + 1)) / 6 - 1;

      //按曲率从小到大冒泡排序
      for (int k = sp + 1; k <= ep; k++) {
        for (int l = k; l >= sp + 1; l--) {
            //如果后面曲率点大于前面，则交换
          if (cloudCurvature[cloudSortInd[l]] < cloudCurvature[cloudSortInd[l - 1]]) {
            int temp = cloudSortInd[l - 1];
            cloudSortInd[l - 1] = cloudSortInd[l];
            cloudSortInd[l] = temp;
          }
        }
      }

      //挑选每个分段的曲率很大和比较大的点
      int largestPickedNum = 0;
      for (int k = ep; k >= sp; k--) {
        int ind = cloudSortInd[k];  //曲率最大点的点序

        //如果曲率大的点，曲率的确比较大，并且未被筛选过滤掉
        if (cloudNeighborPicked[ind] == 0 &&
            cloudCurvature[ind] > 0.1) {
        
          largestPickedNum++;
          if (largestPickedNum <= 2) {//挑选曲率最大的前2个点放入sharp点集合
            cloudLabel[ind] = 2;//2代表点曲率很大
            cornerPointsSharp.push_back(laserCloud->points[ind]);
            cornerPointsLessSharp.push_back(laserCloud->points[ind]);
          } else if (largestPickedNum <= 20) {//挑选曲率最大的前20个点放入less sharp点集合
            cloudLabel[ind] = 1;//1代表点曲率比较尖锐
            cornerPointsLessSharp.push_back(laserCloud->points[ind]);
          } else {
            break;
          }

          cloudNeighborPicked[ind] = 1;//筛选标志置位

          //将曲率比较大的点的前后各5个连续距离比较近的点筛选出去，防止特征点聚集，使得特征点在每个方向上尽量分布均匀
          for (int l = 1; l <= 5; l++) {
            float diffX = laserCloud->points[ind + l].x 
                        - laserCloud->points[ind + l - 1].x;
            float diffY = laserCloud->points[ind + l].y 
                        - laserCloud->points[ind + l - 1].y;
            float diffZ = laserCloud->points[ind + l].z 
                        - laserCloud->points[ind + l - 1].z;
            if (diffX * diffX + diffY * diffY + diffZ * diffZ > 0.05) {
              break;
            }

            cloudNeighborPicked[ind + l] = 1;
          }
          for (int l = -1; l >= -5; l--) {
            float diffX = laserCloud->points[ind + l].x 
                        - laserCloud->points[ind + l + 1].x;
            float diffY = laserCloud->points[ind + l].y 
                        - laserCloud->points[ind + l + 1].y;
            float diffZ = laserCloud->points[ind + l].z 
                        - laserCloud->points[ind + l + 1].z;
            if (diffX * diffX + diffY * diffY + diffZ * diffZ > 0.05) {
              break;
            }

            cloudNeighborPicked[ind + l] = 1;
          }
        }
      }

      //挑选每个分段的曲率很小比较小的点
      int smallestPickedNum = 0;
      for (int k = sp; k <= ep; k++) {
        int ind = cloudSortInd[k];

        //如果曲率的确比较小，并且未被筛选出
        if (cloudNeighborPicked[ind] == 0 &&
            cloudCurvature[ind] < 0.1) {

          cloudLabel[ind] = -1;//-1代表曲率很小的点
          surfPointsFlat.push_back(laserCloud->points[ind]);

          smallestPickedNum++;
          if (smallestPickedNum >= 4) {//只选最小的四个，剩下的Label==0,就都是曲率比较小的
            break;
          }

          cloudNeighborPicked[ind] = 1;
          for (int l = 1; l <= 5; l++) {//同样防止特征点聚集
            float diffX = laserCloud->points[ind + l].x 
                        - laserCloud->points[ind + l - 1].x;
            float diffY = laserCloud->points[ind + l].y 
                        - laserCloud->points[ind + l - 1].y;
            float diffZ = laserCloud->points[ind + l].z 
                        - laserCloud->points[ind + l - 1].z;
            if (diffX * diffX + diffY * diffY + diffZ * diffZ > 0.05) {
              break;
            }

            cloudNeighborPicked[ind + l] = 1;
          }
          for (int l = -1; l >= -5; l--) {
            float diffX = laserCloud->points[ind + l].x 
                        - laserCloud->points[ind + l + 1].x;
            float diffY = laserCloud->points[ind + l].y 
                        - laserCloud->points[ind + l + 1].y;
            float diffZ = laserCloud->points[ind + l].z 
                        - laserCloud->points[ind + l + 1].z;
            if (diffX * diffX + diffY * diffY + diffZ * diffZ > 0.05) {
              break;
            }

            cloudNeighborPicked[ind + l] = 1;
          }
        }
      }

      //将剩余的点（包括之前被排除的点）全部归入平面点中less flat类别中
      for (int k = sp; k <= ep; k++) {
        if (cloudLabel[k] <= 0) {
          surfPointsLessFlatScan->push_back(laserCloud->points[k]);
        }
      }
    }

    //由于less flat点最多，对每个分段less flat的点进行体素栅格滤波
    pcl::PointCloud<PointType> surfPointsLessFlatScanDS;
    pcl::VoxelGrid<PointType> downSizeFilter;
    downSizeFilter.setInputCloud(surfPointsLessFlatScan);
    downSizeFilter.setLeafSize(0.2, 0.2, 0.2);
    downSizeFilter.filter(surfPointsLessFlatScanDS);

    //less flat点汇总
    surfPointsLessFlat += surfPointsLessFlatScanDS;
  }

  //publich消除非匀速运动畸变后的所有的点
  sensor_msgs::PointCloud2 laserCloudOutMsg;
  pcl::toROSMsg(*laserCloud, laserCloudOutMsg);
  laserCloudOutMsg.header.stamp = laserCloudMsg->header.stamp;
  laserCloudOutMsg.header.frame_id = "/camera";
  pubLaserCloud.publish(laserCloudOutMsg);

  //publich消除非匀速运动畸变后的平面点和边沿点
  sensor_msgs::PointCloud2 cornerPointsSharpMsg;
  pcl::toROSMsg(cornerPointsSharp, cornerPointsSharpMsg);
  cornerPointsSharpMsg.header.stamp = laserCloudMsg->header.stamp;
  cornerPointsSharpMsg.header.frame_id = "/camera";
  pubCornerPointsSharp.publish(cornerPointsSharpMsg);

  sensor_msgs::PointCloud2 cornerPointsLessSharpMsg;
  pcl::toROSMsg(cornerPointsLessSharp, cornerPointsLessSharpMsg);
  cornerPointsLessSharpMsg.header.stamp = laserCloudMsg->header.stamp;
  cornerPointsLessSharpMsg.header.frame_id = "/camera";
  pubCornerPointsLessSharp.publish(cornerPointsLessSharpMsg);

  sensor_msgs::PointCloud2 surfPointsFlat2;
  pcl::toROSMsg(surfPointsFlat, surfPointsFlat2);
  surfPointsFlat2.header.stamp = laserCloudMsg->header.stamp;
  surfPointsFlat2.header.frame_id = "/camera";
  pubSurfPointsFlat.publish(surfPointsFlat2);

  sensor_msgs::PointCloud2 surfPointsLessFlat2;
  pcl::toROSMsg(surfPointsLessFlat, surfPointsLessFlat2);
  surfPointsLessFlat2.header.stamp = laserCloudMsg->header.stamp;
  surfPointsLessFlat2.header.frame_id = "/camera";
  pubSurfPointsLessFlat.publish(surfPointsLessFlat2);

  //publich IMU消息,由于循环到了最后，因此是Cur都是代表最后一个点，即最后一个点的欧拉角，畸变位移及一个点云周期增加的速度
  pcl::PointCloud<pcl::PointXYZ> imuTrans(4, 1);
  //起始点欧拉角
  imuTrans.points[0].x = imuPitchStart;
  imuTrans.points[0].y = imuYawStart;
  imuTrans.points[0].z = imuRollStart;

  //最后一个点的欧拉角
  imuTrans.points[1].x = imuPitchCur;
  imuTrans.points[1].y = imuYawCur;
  imuTrans.points[1].z = imuRollCur;

  //最后一个点相对于第一个点的畸变位移和速度
  imuTrans.points[2].x = imuShiftFromStartXCur;
  imuTrans.points[2].y = imuShiftFromStartYCur;
  imuTrans.points[2].z = imuShiftFromStartZCur;

  imuTrans.points[3].x = imuVeloFromStartXCur;
  imuTrans.points[3].y = imuVeloFromStartYCur;
  imuTrans.points[3].z = imuVeloFromStartZCur;

  sensor_msgs::PointCloud2 imuTransMsg;
  pcl::toROSMsg(imuTrans, imuTransMsg);
  imuTransMsg.header.stamp = laserCloudMsg->header.stamp;
  imuTransMsg.header.frame_id = "/camera";
  pubImuTrans.publish(imuTransMsg);
}

//接收imu消息，imu坐标系为x轴向前，y轴向右，z轴向上的右手坐标系
void imuHandler(const sensor_msgs::Imu::ConstPtr& imuIn)
{
  double roll, pitch, yaw;
  tf::Quaternion orientation;
  //convert Quaternion msg to Quaternion
  tf::quaternionMsgToTF(imuIn->orientation, orientation);
  //This will get the roll pitch and yaw from the matrix about fixed axes X, Y, Z respectively. That's R = Rz(yaw)*Ry(pitch)*Rx(roll).
  //Here roll pitch yaw is in the global frame
  tf::Matrix3x3(orientation).getRPY(roll, pitch, yaw);

  //减去重力的影响,求出xyz方向的加速度实际值，并进行坐标轴交换，统一到z轴向前,x轴向左的右手坐标系, 交换过后RPY对应fixed axes ZXY(RPY---ZXY)。Now R = Ry(yaw)*Rx(pitch)*Rz(roll).
  float accX = imuIn->linear_acceleration.y - sin(roll) * cos(pitch) * 9.81;
  float accY = imuIn->linear_acceleration.z - cos(roll) * cos(pitch) * 9.81;
  float accZ = imuIn->linear_acceleration.x + sin(pitch) * 9.81;

  //循环移位效果，形成环形数组
  imuPointerLast = (imuPointerLast + 1) % imuQueLength;

  imuTime[imuPointerLast] = imuIn->header.stamp.toSec();
  imuRoll[imuPointerLast] = roll;
  imuPitch[imuPointerLast] = pitch;
  imuYaw[imuPointerLast] = yaw;
  imuAccX[imuPointerLast] = accX;
  imuAccY[imuPointerLast] = accY;
  imuAccZ[imuPointerLast] = accZ;

  AccumulateIMUShift();
}

int main(int argc, char** argv)
{
  ros::init(argc, argv, "scanRegistration");
  ros::NodeHandle nh;

  ros::Subscriber subLaserCloud = nh.subscribe<sensor_msgs::PointCloud2> 
                                  ("/velodyne_points", 2, laserCloudHandler);

  ros::Subscriber subImu = nh.subscribe<sensor_msgs::Imu> ("/imu/data", 50, imuHandler);

  pubLaserCloud = nh.advertise<sensor_msgs::PointCloud2>
                                 ("/velodyne_cloud_2", 2);

  pubCornerPointsSharp = nh.advertise<sensor_msgs::PointCloud2>
                                        ("/laser_cloud_sharp", 2);

  pubCornerPointsLessSharp = nh.advertise<sensor_msgs::PointCloud2>
                                            ("/laser_cloud_less_sharp", 2);

  pubSurfPointsFlat = nh.advertise<sensor_msgs::PointCloud2>
                                       ("/laser_cloud_flat", 2);

  pubSurfPointsLessFlat = nh.advertise<sensor_msgs::PointCloud2>
                                           ("/laser_cloud_less_flat", 2);

  pubImuTrans = nh.advertise<sensor_msgs::PointCloud2> ("/imu_trans", 5);

  ros::spin();

  return 0;
}

