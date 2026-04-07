using System;
using System.Collections;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using TMPro;
using UnityEngine;
using UnityEngine.Networking;
using NeuroXess;

[Serializable]
public class EEGPoint
{
    public int ch;
    public float val;
    public string sTime;
}

[Serializable]
public class DataWrapper
{
    public List<EEGPoint> items = new List<EEGPoint>();
    public int user_id = 1;
    public string event_label;
}

[Serializable]
public class OutboundBatch
{
    public DataWrapper payload;
    public int sampleCount;
    public float createdAt;
}

public class NewNeuroManagerService : MonoBehaviour
{
    private const int MinTrackedChannel = 0;
    private const int MaxTrackedChannel = 3;

    private NXBTManager m_NXBTManager;
    private NXBCIManager m_NXBCIManager;
    private NXBreathManager m_NXBreathManager;
    private NXPPGManager m_NXPPGManager;
    private NXAuthManager m_NXAuthManager;
    private readonly List<byte[]> mBluetoothDeviceAddress = new List<byte[]>();

    [Header("UI")]
    public TMP_Text tmpText;

    [Header("Backend")]
    [SerializeField] private string backendUploadUrl = "http://192.168.0.162:8000/api/eeg/upload";
    [SerializeField] private int userId = 1;
    [SerializeField] private int sampleRateHz = 250;
    [SerializeField] private float flushIntervalSeconds = 0.10f;
    [SerializeField] private int maxBatchSize = 256;
    [SerializeField] private float retryDelaySeconds = 1.0f;

    [Header("Recording")]
    [SerializeField] private bool recordToCsv;

    private readonly object bufferLock = new object();
    private readonly List<EEGPoint> pendingPoints = new List<EEGPoint>(1024);
    private readonly Queue<OutboundBatch> outboundQueue = new Queue<OutboundBatch>();
    private readonly Dictionary<int, long> channelSampleCounts = new Dictionary<int, long>();

    private bool isRecording;
    private bool isSending;
    private DateTime? streamStartUtc;
    private StreamWriter csvWriter;
    private string filePath;

    void Start()
    {
        m_NXBTManager = new NXBTManager();
        m_NXBCIManager = new NXBCIManager();
        m_NXBreathManager = new NXBreathManager();
        m_NXPPGManager = new NXPPGManager();
        m_NXAuthManager = new NXAuthManager();

        m_NXBTManager.addOnConnectEvent(BT_onConnectEvent);
        m_NXBTManager.addOnDeviceEvent(BT_onDeviceEvent);
        m_NXBTManager.addOnScanDeviceEvent(BT_onScanDevice);
        m_NXBTManager.enableBTPermission(true);

        m_NXBCIManager.addOnBandPowerEvent(BCI_onBandPowerEvent);
        m_NXBCIManager.addOnRawDataEvent(BCI_onRawDataEvent);
        m_NXBCIManager.addOnPracticeEvent(BCI_onPracticeEvent);

        m_NXBreathManager.addOnBreathEvent(BRT_onBreathEvent);
        m_NXBreathManager.addOnRawDataEvent(BRT_onRawDataEvent);

        m_NXPPGManager.addOnHREvent(HRT_onHeartRate);
        m_NXPPGManager.addOnRawDataEvent(HRT_onRawDataEvent);
        m_NXPPGManager.addOnHRVRMSSDEvent(HRT_onHRVRMSSDEvent);

        m_NXAuthManager.setAuthorizeCode("87e226da-2415-4b94-8047-db7ae18b95e8");
        m_NXAuthManager.addOnAuthorizeEvent(AUTH_OnAuthorizeEvent);

        NXManager.GetInstance().initSDK(
            m_NXBTManager,
            m_NXBCIManager,
            m_NXBreathManager,
            m_NXPPGManager,
            m_NXAuthManager
        );

        InvokeRepeating(nameof(updateBreathData), 0.1f, 0.033f);
        InvokeRepeating(nameof(FlushPendingSamples), flushIntervalSeconds, flushIntervalSeconds);
    }

    void updateBreathData()
    {
        NXManager.GetInstance().updateBreathData();
    }

    void OnDestroy()
    {
        CancelInvoke(nameof(updateBreathData));
        CancelInvoke(nameof(FlushPendingSamples));

        FlushPendingSamples();
        StopRecordingData();

        NXManager.GetInstance().deInitSDK();
    }

    public void BT_onConnectEvent(NXBTManager.BTConnectEvent status, byte[] message)
    {
        if (NXBTManager.BTConnectEvent.EVENT_BT_SCAN_STOP == status && mBluetoothDeviceAddress.Count > 0)
        {
            NXBTManager.connectDevice(mBluetoothDeviceAddress.First());
        }
    }

    public void BT_onDeviceEvent(NXBTManager.BTDeviceEvent status, byte[] value)
    {
    }

    public void BT_onScanDevice(byte[] mac, byte[] name)
    {
        string macStr = BitConverter.ToString(mac);
        Debug.Log($"[Neuro] discovered device: {macStr}");

        bool exists = mBluetoothDeviceAddress.Any(address => address.SequenceEqual(mac));
        if (!exists)
        {
            mBluetoothDeviceAddress.Add(mac);
        }
    }

    public void BCI_onBandPowerEvent(double[] data)
    {
    }

    public void BCI_onRawDataEvent(double[] data, int channel)
    {
        if (data == null || data.Length == 0)
        {
            return;
        }

        if (channel < MinTrackedChannel || channel > MaxTrackedChannel)
        {
            return;
        }

        EnqueueSamples(channel, data);

        double latestValue = data[data.Length - 1];
        Debug.Log($"[NeuroXess] channel {channel}, samples {data.Length}, latest {latestValue}");
        if (tmpText != null)
        {
            tmpText.text = $"ch {channel} latest {latestValue:F3}";
        }

        if (isRecording)
        {
            if (tmpText != null)
            {
                tmpText.text = $"ch {channel} latest {latestValue:F3}";
            }
        }
    }

    private void EnqueueSamples(int channel, double[] samples)
    {
        // --- 开关逻辑 ---
        if (!isRecording) return;
        double sampleIntervalMs = 1000.0 / Mathf.Max(1, sampleRateHz);

        lock (bufferLock)
        {
            if (!streamStartUtc.HasValue)
            {
                streamStartUtc = DateTime.UtcNow.AddMilliseconds(-(samples.Length - 1) * sampleIntervalMs);
            }

            long startIndex = channelSampleCounts.TryGetValue(channel, out long currentIndex) ? currentIndex : 0;

            for (int index = 0; index < samples.Length; index++)
            {
                DateTime sampleTime = streamStartUtc.Value.AddMilliseconds((startIndex + index) * sampleIntervalMs);
                EEGPoint point = new EEGPoint
                {
                    ch = channel,
                    val = (float)samples[index],
                    sTime = sampleTime.ToString("yyyy-MM-ddTHH:mm:ss.fffZ")
                };

                pendingPoints.Add(point);

                if (recordToCsv && isRecording && csvWriter != null)
                {
                    csvWriter.WriteLine($"{point.sTime},{channel},{point.val}");
                }
            }

            channelSampleCounts[channel] = startIndex + samples.Length;
        }
    }

    private void FlushPendingSamples()
    {
        List<EEGPoint> snapshot;

        lock (bufferLock)
        {
            if (pendingPoints.Count == 0)
            {
                return;
            }

            snapshot = new List<EEGPoint>(pendingPoints);
            pendingPoints.Clear();
        }

        int offset = 0;
        while (offset < snapshot.Count)
        {
            int batchCount = Mathf.Min(maxBatchSize, snapshot.Count - offset);
            DataWrapper wrapper = new DataWrapper
            {
                user_id = userId,
                items = snapshot.GetRange(offset, batchCount)
            };

            outboundQueue.Enqueue(new OutboundBatch
            {
                payload = wrapper,
                sampleCount = batchCount,
                createdAt = Time.unscaledTime
            });
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
            OutboundBatch batch = outboundQueue.Dequeue();
            string json = JsonUtility.ToJson(batch.payload);
            byte[] requestBody = System.Text.Encoding.UTF8.GetBytes(json);
            Debug.Log($"[NeuroXessLog] JSON Context: {json}");
            using (UnityWebRequest request = new UnityWebRequest(backendUploadUrl, UnityWebRequest.kHttpVerbPOST))
            {
                request.uploadHandler = new UploadHandlerRaw(requestBody);
                request.downloadHandler = new DownloadHandlerBuffer();
                request.SetRequestHeader("Content-Type", "application/json");

                yield return request.SendWebRequest();

                if (request.result == UnityWebRequest.Result.Success)
                {
                    Debug.Log($"[NeuroXess] uploaded {batch.sampleCount} points to backend");
                }
                else
                {
                    Debug.LogWarning($"[NeuroXess] upload failed: {request.error}");

                    if (request.downloadHandler != null)
                    {
                        Debug.LogError($"[NeuroXessLog] error detail: {request.downloadHandler.text}");
                    }
                    outboundQueue.Enqueue(batch);
                    yield return new WaitForSecondsRealtime(retryDelaySeconds);
                }
            }
        }

        isSending = false;
    }

    public void BCI_onPracticeEvent(NXBCIManager.BCIPracticeEvent status, double[] value)
    {
        switch (status)
        {
            case NXBCIManager.BCIPracticeEvent.EVENT_PRACTICE_CLOSED_EYES:
                break;
            case NXBCIManager.BCIPracticeEvent.EVENT_PRACTICE_CURRENT_SCORE:
                break;
        }
    }

    public void BRT_onBreathEvent(double a, int b, int c, int d)
    {
    }

    public void BRT_onRawDataEvent(double[] data, int channel)
    {
    }

    public void HRT_onHeartRate(int t)
    {
    }

    public void HRT_onRawDataEvent(double[] data, int channel)
    {
    }

    public void HRT_onHRVRMSSDEvent(double rmssd)
    {
    }

    public void AUTH_OnAuthorizeEvent(NXAuthManager.AuthorizeEvent events, byte[] msg)
    {
        Debug.Log($"[NeuroXess] auth state: {events}");
    }

    public void StartRecordingData()
    {
        if (isRecording)
        {
            return;
        }
        // 1. 重置同步状态，确保新一轮录制的时间戳从 0 开始重新校准
        lock (bufferLock)
        {
            streamStartUtc = null;
            channelSampleCounts.Clear();
            pendingPoints.Clear();
        }
        string timeIntStr = DateTime.Now.ToString("MMddHHmm");
        // 将这个字符串转换成整数，赋值给 userId
        userId = int.Parse(timeIntStr);
        Debug.Log($"[NeuroXess]  UserID : {userId}");

        string timeStamp = DateTime.Now.ToString("yyyyMMdd_HHmmss");
        filePath = Path.Combine(Application.persistentDataPath, $"EEG_Data_{timeStamp}.csv");

        try
        {
            csvWriter = new StreamWriter(filePath, true);
            csvWriter.WriteLine("Timestamp,Channel,DataValue");
            isRecording = true;
            Debug.Log($"[NeuroXess] start recording, saved path: {filePath}");
            if (tmpText != null)
            {
                tmpText.text = "state: start recording";
            }
        }
        catch (Exception e)
        {
            Debug.LogError($"[NeuroXess] file create failed: {e.Message}");
        }
    }

    public void StopRecordingData()
    {
        if (!isRecording)
        {
            return;
        }

        isRecording = false;
        FlushPendingSamples();

        if (csvWriter != null)
        {
            csvWriter.Flush();
            csvWriter.Close();
            csvWriter = null;
        }

        Debug.Log("[NeuroXess] stop recording, data saved.");
        if (tmpText != null)
        {
            tmpText.text = "state: stop recording";
        }
    }
}
