from enum import IntEnum


# the D400 firmware notification codes and texts, from librealsense v2.58.3 src/ds/d400/d400-private.h:
# enum d400_notifications_types and std::map d400_fw_error_report
class D400NotificationType(IntEnum):
    SUCCESS = 0
    HOT_LASER_POWER_REDUCE = 1
    HOT_LASER_DISABLE = 2
    FLAG_B_LASER_DISABLE = 3
    STEREO_MODULE_NOT_CONNECTED = 4
    EEPROM_CORRUPTED = 5
    CALIBRATION_CORRUPTED = 6
    MM_UPD_FAIL = 7
    ISP_UPD_FAIL = 8
    MM_FORCE_PAUSE = 9
    MM_FAILURE = 10
    USB_SCP_OVERFLOW = 11
    USB_REC_OVERFLOW = 12
    USB_CAM_OVERFLOW = 13
    MIPI_LEFT_ERROR = 14
    MIPI_RIGHT_ERROR = 15
    MIPI_RT_ERROR = 16
    MIPI_FE_ERROR = 17
    I2C_CFG_LEFT_ERROR = 18
    I2C_CFG_RIGHT_ERROR = 19
    I2C_CFG_RT_ERROR = 20
    I2C_CFG_FE_ERROR = 21
    STREAM_NOT_START_Z = 22
    STREAM_NOT_START_Y = 23
    STREAM_NOT_START_CAM = 24
    REC_ERROR = 25
    USB2_LIMIT = 26
    COLD_LASER_DISABLE = 27
    NO_TEMPERATURE_DISABLE_LASER = 28
    ISP_BOOT_DATA_UPLOAD_FAILED = 29


# the texts are the same as in librealsense, including the "IC2" spelling
D400_FW_ERROR_REPORT = {
    D400NotificationType.SUCCESS: "Success",
    D400NotificationType.HOT_LASER_POWER_REDUCE: "Laser hot - power reduce",
    D400NotificationType.HOT_LASER_DISABLE: "Laser hot - disabled",
    D400NotificationType.FLAG_B_LASER_DISABLE: "Flag B - laser disabled",
    D400NotificationType.STEREO_MODULE_NOT_CONNECTED: "Stereo Module is not connected",
    D400NotificationType.EEPROM_CORRUPTED: "EEPROM corrupted",
    D400NotificationType.CALIBRATION_CORRUPTED: "Calibration corrupted",
    D400NotificationType.MM_UPD_FAIL: "Motion Module update failed",
    D400NotificationType.ISP_UPD_FAIL: "ISP update failed",
    D400NotificationType.MM_FORCE_PAUSE: "Motion Module force pause",
    D400NotificationType.MM_FAILURE: "Motion Module failure",
    D400NotificationType.USB_SCP_OVERFLOW: "USB SCP overflow",
    D400NotificationType.USB_REC_OVERFLOW: "USB REC overflow",
    D400NotificationType.USB_CAM_OVERFLOW: "USB CAM overflow",
    D400NotificationType.MIPI_LEFT_ERROR: "Left MIPI error",
    D400NotificationType.MIPI_RIGHT_ERROR: "Right MIPI error",
    D400NotificationType.MIPI_RT_ERROR: "RT MIPI error",
    D400NotificationType.MIPI_FE_ERROR: "FishEye MIPI error",
    D400NotificationType.I2C_CFG_LEFT_ERROR: "Left IC2 Config error",
    D400NotificationType.I2C_CFG_RIGHT_ERROR: "Right IC2 Config error",
    D400NotificationType.I2C_CFG_RT_ERROR: "RT IC2 Config error",
    D400NotificationType.I2C_CFG_FE_ERROR: "FishEye IC2 Config error",
    D400NotificationType.STREAM_NOT_START_Z: "Depth stream start failure",
    D400NotificationType.STREAM_NOT_START_Y: "IR stream start failure",
    D400NotificationType.STREAM_NOT_START_CAM: "Camera stream start failure",
    D400NotificationType.REC_ERROR: "REC error",
    D400NotificationType.USB2_LIMIT: "USB2 Limit",
    D400NotificationType.COLD_LASER_DISABLE: "Laser cold - disabled",
    D400NotificationType.NO_TEMPERATURE_DISABLE_LASER: "Temperature read failure - laser disabled",
    D400NotificationType.ISP_BOOT_DATA_UPLOAD_FAILED: "ISP boot data upload failure",
}


# what to do during a recording, only for the codes with a clear action
# this advice is general camera practice, not from librealsense
USB_OVERFLOW = "The computer does not read data fast enough. Close other programs, or use a lower resolution or frame rate."
MIPI_ERROR = (
    "Stop and start the stream again. If the error repeats, disconnect and connect the camera. If it still repeats, the camera needs service."
)
STREAM_START_FAILURE = "Close other programs that use the camera, such as Intel RealSense Viewer. Then start the stream again."

D400_WHAT_TO_DO = {
    D400NotificationType.HOT_LASER_POWER_REDUCE: (
        "Let the camera cool. Keep it out of direct sun and away from heat sources. Improve the airflow."
    ),
    D400NotificationType.HOT_LASER_DISABLE: (
        "Stop the recording. Let the camera cool, then start again. The depth data has no laser pattern until then."
    ),
    D400NotificationType.COLD_LASER_DISABLE: "Let the camera warm up to room temperature, then start again.",
    D400NotificationType.STEREO_MODULE_NOT_CONNECTED: (
        "Disconnect and connect the camera again. If the error repeats, the camera needs service."
    ),
    D400NotificationType.CALIBRATION_CORRUPTED: "Do not use the depth data. Calibrate the camera again in Intel RealSense Viewer.",
    D400NotificationType.USB_SCP_OVERFLOW: USB_OVERFLOW,
    D400NotificationType.USB_REC_OVERFLOW: USB_OVERFLOW,
    D400NotificationType.USB_CAM_OVERFLOW: USB_OVERFLOW,
    D400NotificationType.MIPI_LEFT_ERROR: MIPI_ERROR,
    D400NotificationType.MIPI_RIGHT_ERROR: MIPI_ERROR,
    D400NotificationType.STREAM_NOT_START_Z: STREAM_START_FAILURE,
    D400NotificationType.STREAM_NOT_START_Y: STREAM_START_FAILURE,
    D400NotificationType.STREAM_NOT_START_CAM: STREAM_START_FAILURE,
    D400NotificationType.USB2_LIMIT: "Connect the camera to a USB 3 port with a USB 3 cable.",
}
