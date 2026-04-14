using System.IO;
using Unity.XR.PXR;
using UnityEngine;

public class EyeDataLogger : MonoBehaviour
{
    // ================== 视线追踪状态变量 ==================
    private TrackingStateCode trackingState;
    private bool isSupportedEyeTracking = false;

    // ================== 数据保存相关变量 ==================
    private string gazeSavePath;
    private StreamWriter gazeCsvWriter;

    private string blinkSavePath;
    private StreamWriter blinkCsvWriter;

    private bool isWriting = false;

    private void Awake()
    {
        // 1. 请求眼动追踪服务
        PXR_MotionTracking.WantEyeTrackingService();
        Debug.Log("[EyeDataLogger] eye tracking service requested.");
    }

    private void Start()
    {
        // 2. 开启眼动追踪
        EyeTrackingStartInfo info = new EyeTrackingStartInfo();
        info.needCalibration = 1;
        info.mode = EyeTrackingMode.PXR_ETM_BOTH;
        trackingState = (TrackingStateCode)PXR_MotionTracking.StartEyeTracking(ref info);
        isSupportedEyeTracking = trackingState == TrackingStateCode.PXR_MT_SUCCESS;

        if (!isSupportedEyeTracking)
        {
            Debug.LogWarning($"[EyeDataLogger] eye tracking start failed: {trackingState}");
        }

        // 3. 初始化数据保存文件 (.csv格式)
        gazeSavePath = Path.Combine(Application.persistentDataPath, "EyeTrackingData.csv");
        blinkSavePath = Path.Combine(Application.persistentDataPath, "EyeBlinkData.csv");

        try
        {
            // 初始化视线数据流
            gazeCsvWriter = new StreamWriter(gazeSavePath, false);
            gazeCsvWriter.WriteLine("Timestamp,PosX,PosY,PosZ,OriX,OriY,OriZ,OriW");

            // 初始化眨眼数据流
            blinkCsvWriter = new StreamWriter(blinkSavePath, false);
            blinkCsvWriter.WriteLine("Timestamp_ns,IsLeftBlink,IsRightBlink");

            isWriting = true;
            Debug.Log($"[EyeDataLogger] start recorfing.\n eyepath file: {gazeSavePath}\n eyeblink file: {blinkSavePath}");
        }
        catch (System.Exception e)
        {
            Debug.LogError($"[EyeDataLogger] create file failed: {e.Message}");
            isWriting = false;
        }
    }

    private void FixedUpdate()
    {
        if (isSupportedEyeTracking && isWriting)
        {
            // ================= 记录视线追踪数据 =================
            EyeTrackingDataGetInfo info = new EyeTrackingDataGetInfo();
            info.displayTime = 0;
            info.flags = EyeTrackingDataGetFlags.PXR_EYE_DEFAULT
                       | EyeTrackingDataGetFlags.PXR_EYE_POSITION
                       | EyeTrackingDataGetFlags.PXR_EYE_ORIENTATION;

            EyeTrackingData eyeTrackingData = new EyeTrackingData();
            trackingState = (TrackingStateCode)PXR_MotionTracking.GetEyeTrackingData(ref info, ref eyeTrackingData);

            if (trackingState == TrackingStateCode.PXR_MT_SUCCESS)
            {
                var pose = eyeTrackingData.eyeDatas[2].pose;
                string gazeDataLine = $"{Time.realtimeSinceStartup}," +
                                      $"{pose.position.x},{pose.position.y},{pose.position.z}," +
                                      $"{pose.orientation.x},{pose.orientation.y},{pose.orientation.z},{pose.orientation.w}";
                gazeCsvWriter.WriteLine(gazeDataLine);
            }

            // ================= 记录眨眼数据 =================
            long blinkTimestamp = 0;
            bool isLeftBlink = false;
            bool isRightBlink = false;

            // 调用获取眨眼数据的 API
            int blinkStatus = PXR_MotionTracking.GetEyeBlink(ref blinkTimestamp, ref isLeftBlink, ref isRightBlink);

            // 返回值为 0 表示获取成功
            if (blinkStatus == 0)
            {
                // 将 bool 转换为 1 或 0 方便 CSV 记录和后续数据分析 (或者直接写 true/false)
                int leftBlinkVal = isLeftBlink ? 1 : 0;
                int rightBlinkVal = isRightBlink ? 1 : 0;

                string blinkDataLine = $"{blinkTimestamp},{leftBlinkVal},{rightBlinkVal}";
                blinkCsvWriter.WriteLine(blinkDataLine);
            }
        }
    }

    private void OnDestroy()
    {
        // 4. 关闭眼动追踪并释放文件流
        if (isSupportedEyeTracking)
        {
            EyeTrackingStopInfo info = new EyeTrackingStopInfo();
            trackingState = (TrackingStateCode)PXR_MotionTracking.StopEyeTracking(ref info);
        }

        // 关闭视线数据文件
        if (gazeCsvWriter != null)
        {
            gazeCsvWriter.Flush();
            gazeCsvWriter.Close();
            gazeCsvWriter.Dispose();
        }

        // 关闭眨眼数据文件
        if (blinkCsvWriter != null)
        {
            blinkCsvWriter.Flush();
            blinkCsvWriter.Close();
            blinkCsvWriter.Dispose();
        }

        if (isWriting)
        {
            isWriting = false;
            Debug.Log("[EyeDataLogger] all file saved。");
        }
    }
}
