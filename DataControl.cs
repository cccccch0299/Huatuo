using System;
using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using UnityEngine.Networking;
using UnityEngine.UI;
using Unity.XR.PXR;

public enum gameState
{
    Start,
    Pause,
    ReStart,
    End,
    Knock,
    Null
}

public class DataControl : MonoBehaviour
{
    public Text showText = null;

    [Header("Backend")]
    [SerializeField]
    private string path = "http://124.71.201.126:5000/api/eeg/upload";
    [SerializeField] private int sampleRateHz = 250;
    [SerializeField] private float flushIntervalSeconds = 0.10f;
    [SerializeField] private int maxBatchSize = 256;
    [SerializeField] private float retryDelaySeconds = 1.0f;

    private bool support = false;
    private EyeTrackingMode[] eyeTrackingModes;
    private EyeTrackingState eyeTrackingState = new EyeTrackingState();
    private float[] currentEyeData = new float[3];
    private float leftEyeOpenness = 0f;
    private float rightEyeOpenness = 0f;

    private readonly object bufferLock = new object();
    private readonly List<MixedData> pendingPoints = new List<MixedData>(1024);
    private readonly Queue<MixedDatas> outboundQueue = new Queue<MixedDatas>();
    private readonly Dictionary<int, long> channelSampleCounts = new Dictionary<int, long>();

    private bool isSending;
    private DateTime streamStartUtc;

    private int currentTime = 111111;
    private gameState currentState = gameState.Null;
    private Coroutine stateCoroutine = null;

    private void Start()
    {
    	currentTime = DateTime.Now.Month * 1000000 + DateTime.Now.Day * 10000 +
                     DateTime.Now.Hour * 100 + DateTime.Now.Minute;
        streamStartUtc = DateTime.UtcNow;
        InitializeEyeTracking();
        InvokeRepeating(nameof(FlushPendingSamples), flushIntervalSeconds, flushIntervalSeconds);
    }

    private void InitializeEyeTracking()
    {
        int supportModesCount = 0;
        PXR_MotionTracking.GetEyeTrackingSupported(ref support, ref supportModesCount, ref eyeTrackingModes);

        if (support)
        {
            EyeTrackingStartInfo eyeTrackingStartInfo = new EyeTrackingStartInfo();
            eyeTrackingStartInfo.needCalibration = 1;
            eyeTrackingStartInfo.mode = EyeTrackingMode.PXR_ETM_BOTH;
            PXR_MotionTracking.StartEyeTracking(ref eyeTrackingStartInfo);
            Debug.Log("眼动追踪已启动");
        }
        else
        {
            Debug.LogWarning("当前设备不支持眼动追踪");
        }
    }

    private void FixedUpdate()
    {
        

        if (support)
        {
            bool tracking = false;
            EyeTrackingState eyeTrackingState = new EyeTrackingState();
            PXR_MotionTracking.GetEyeTrackingState(ref tracking, ref eyeTrackingState);

            if (tracking)
            {
                EyeTrackingDataGetInfo info = new EyeTrackingDataGetInfo();
                info.displayTime = 0;
                info.flags = EyeTrackingDataGetFlags.PXR_EYE_DEFAULT
                | EyeTrackingDataGetFlags.PXR_EYE_POSITION
                | EyeTrackingDataGetFlags.PXR_EYE_ORIENTATION;
                EyeTrackingData eyeTrackingData = new EyeTrackingData();
                PXR_MotionTracking.GetEyeTrackingData(ref info, ref eyeTrackingData);
                PXR_MotionTracking.GetEyeOpenness(ref leftEyeOpenness, ref rightEyeOpenness);

                var pose = eyeTrackingData.eyeDatas[2].pose;
                var rotation = new Quaternion(-pose.position.x, -pose.position.y, pose.position.z, pose.orientation.x);

                Vector3 dir = rotation * Vector3.forward;
                currentEyeData = new float[3] { dir.x, dir.y, dir.z };

                // 每帧生成精确时间戳（基于 streamStartUtc 与 EEG 数据统一时间基准）
                double eyeElapsedMs = (DateTime.UtcNow - streamStartUtc).TotalMilliseconds;
                string sampleTime = streamStartUtc.AddMilliseconds(eyeElapsedMs).ToString("yyyy-MM-ddTHH:mm:ss.fffZ");

                List<MixedData> eyePoints = new List<MixedData>(5);
                eyePoints.Add(new MixedData(4, leftEyeOpenness, sampleTime));
                eyePoints.Add(new MixedData(5, rightEyeOpenness, sampleTime));
                eyePoints.Add(new MixedData(6, currentEyeData[0], sampleTime));
                eyePoints.Add(new MixedData(7, currentEyeData[1], sampleTime));
                eyePoints.Add(new MixedData(8, currentEyeData[2], sampleTime));

                lock (bufferLock)
                {
                    pendingPoints.AddRange(eyePoints);
                }
            }
        }
    }

    /// <summary>
    /// 外部调用：传入 EEG/EMG 原始采样数据（接口不变）
    /// </summary>
    public void AddEEGData(int channel, double[] data)
    {
        if (data == null || data.Length == 0) return;
        if (channel < 0 || channel > 3) return;

        double sampleIntervalMs = 1000.0 / Mathf.Max(1, sampleRateHz);

        lock (bufferLock)
        {
            long startIndex = channelSampleCounts.TryGetValue(channel, out long currentIndex) ? currentIndex : 0;

            for (int index = 0; index < data.Length; index++)
            {
                DateTime sampleTime = streamStartUtc.AddMilliseconds((startIndex + index) * sampleIntervalMs);
                MixedData point = new MixedData
                {
                    ch = channel,
                    val = data[index],
                    sTime = sampleTime.ToString("yyyy-MM-ddTHH:mm:ss.fffZ")
                };
                pendingPoints.Add(point);
            }

            channelSampleCounts[channel] = startIndex + data.Length;
        }
    }

    public void SetState(gameState gameState)
    {
        if (stateCoroutine != null) { StopCoroutine(stateCoroutine); stateCoroutine = null; }
        currentState = gameState;
        stateCoroutine = StartCoroutine(ResetState());
    }

    IEnumerator ResetState()
    {
        yield return new WaitForSeconds(1f);
        currentState = gameState.Null;
        stateCoroutine = null;
    }

    private void FlushPendingSamples()
    {
        List<MixedData> snapshot;

        lock (bufferLock)
        {
            if (pendingPoints.Count == 0) return;

            snapshot = new List<MixedData>(pendingPoints);
            pendingPoints.Clear();
        }

        // 按 maxBatchSize 分批入队
        int offset = 0;
        while (offset < snapshot.Count)
        {
            int batchCount = Mathf.Min(maxBatchSize, snapshot.Count - offset);
            MixedDatas wrapper = new MixedDatas
            {
                user_id = currentTime,
                event_label = currentState.ToString(),
                items = snapshot.GetRange(offset, batchCount)
            };
            outboundQueue.Enqueue(wrapper);
            offset += batchCount;
        }

        if (!isSending)
        {
            StartCoroutine(SendLoop());
        }
    }

    private IEnumerator SendLoop()
    {
        isSending = true;

        while (outboundQueue.Count > 0)
        {
            MixedDatas data = outboundQueue.Dequeue();
            string json = JsonUtility.ToJson(data);
            byte[] requestBody = System.Text.Encoding.UTF8.GetBytes(json);

            using (UnityWebRequest request = new UnityWebRequest(path, UnityWebRequest.kHttpVerbPOST))
            {
                request.uploadHandler = new UploadHandlerRaw(requestBody);
                request.downloadHandler = new DownloadHandlerBuffer();
                request.SetRequestHeader("Content-Type", "application/json");
                request.timeout = 10;

                yield return request.SendWebRequest();

                if (request.result == UnityWebRequest.Result.Success)
                {
                    if (showText != null) showText.text = $"发送成功: {data.items.Count}点";
                }
                else
                {
                    if (showText != null) showText.text = $"发送失败: {request.error}";
                    // 失败重入队，延时重试
                    outboundQueue.Enqueue(data);
                    yield return new WaitForSecondsRealtime(retryDelaySeconds);
                }
            }
        }

        isSending = false;
    }

    private void OnDisable()
    {
        CancelInvoke(nameof(FlushPendingSamples));

        if (stateCoroutine != null) { StopCoroutine(stateCoroutine); stateCoroutine = null; }

        if (support)
        {
            EyeTrackingStopInfo eyeTrackingStopInfo = new EyeTrackingStopInfo();
            PXR_MotionTracking.StopEyeTracking(ref eyeTrackingStopInfo);
        }

        FlushPendingSamples();
        outboundQueue.Clear();
        isSending = false;
    }
}

[Serializable]
public class MixedDatas
{
    public int user_id;
    public string event_label;
    public List<MixedData> items;
}

[Serializable]
public class MixedData
{
    public int ch;
    public double val;
    public string sTime;

    public MixedData() { }
    public MixedData(int ch, double val, string sTime)
    {
        this.ch = ch;
        this.val = val;
        this.sTime = sTime;
    }
}
