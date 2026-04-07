using UnityEngine;
using UnityEngine.Android;
using System;
using System.Collections.Generic;
using System.Collections.Concurrent;
using UnityEngine.Networking;
using System.Collections;
using System.Linq;
// Step 1,use namespace
using NeuroXess;

using TMPro;
using System.IO;
public class NeuroManagerService : MonoBehaviour
{
    // Step 2,declare these managers 
    private NXBTManager m_NXBTManager;
    private NXBCIManager m_NXBCIManager;
    private NXBreathManager m_NXBreathManager;
    private NXPPGManager m_NXPPGManager;
    private NXAuthManager m_NXAuthManager;
    private List<byte[]> mBluetoothDeviceAddress = new List<byte[]>();

    public TMP_Text tmpText;
    // --- 新增：用于控制记录和保存数据的变量 ---
    private string _latestValueUpdate = "";
    private bool isRecording = false;
    private StreamWriter csvWriter;
    private string filePath;

    // --- 新增：数据库同步相关的变量 ---
    [Serializable]
    public class EEGPoint
    {
        public int ch;      // 通道号 (0-3 是信号, 4 是时间戳)
        public double val;  // 数值
        public string sTime; // 系统接收时间 (作为辅助参考)
    }

    // 线程安全队列，用于存放从 SDK 拿到的所有原始点
    private ConcurrentQueue<EEGPoint> dataQueue = new ConcurrentQueue<EEGPoint>();
    private List<EEGPoint> uploadBatch = new List<EEGPoint>();
    public int batchSize = 500; // 建议设为 500 (大约是 5 个通道共 0.4 秒的数据量)
    // ------------------------------------------

    void Start()
    {
        // Step 3, set callback functions for managers
        m_NXBTManager = new NXBTManager();
        m_NXBCIManager = new NXBCIManager();
        m_NXBreathManager = new NXBreathManager();
        m_NXPPGManager = new NXPPGManager();
        m_NXAuthManager = new NXAuthManager();
        
        m_NXBTManager.addOnConnectEvent(BT_onConnectEvent);
        m_NXBTManager.addOnDeviceEvent(BT_onDeviceEvent);
        m_NXBTManager.addOnScanDeviceEvent(BT_onScanDevice);
        //enable bluetooth permission
        m_NXBTManager.enableBTPermission(true);
        
        m_NXBCIManager.addOnBandPowerEvent(BCI_onBandPowerEvent);
        m_NXBCIManager.addOnRawDataEvent(BCI_onRawDataEvent);
        m_NXBCIManager.addOnPracticeEvent(BCI_onPracticeEvent);

        m_NXBreathManager.addOnBreathEvent(BRT_onBreathEvent);
        m_NXBreathManager.addOnRawDataEvent(BRT_onRawDataEvent);

        m_NXPPGManager.addOnHREvent(HRT_onHeartRate);
        m_NXPPGManager.addOnRawDataEvent(HRT_onRawDataEvent);
        m_NXPPGManager.addOnHRVRMSSDEvent(HRT_onHRVRMSSDEvent);

        //Must input authorization code before use SDK to develop your application!
        m_NXAuthManager.setAuthorizeCode("87e226da-2415-4b94-8047-db7ae18b95e8");
        m_NXAuthManager.addOnAuthorizeEvent(AUTH_OnAuthorizeEvent);

        // Step 4, init SDK
        NXManager.GetInstance().initSDK(m_NXBTManager,m_NXBCIManager,m_NXBreathManager,m_NXPPGManager, m_NXAuthManager);



        // --- 新加的代码：请求权限并启动扫描 ---
        Debug.Log("[Neuro] 手动尝试启动蓝牙扫描...");
        m_NXBTManager.enableBTPermission(true); // 确保权限开启
        // ------------------------------------


        // Step 5, must repeating call NXManager.GetInstance().updateBreathData() to start engine of breath,otherwise it will not work.
        // Set repeat rate as 0.033f is important！ 
        InvokeRepeating("updateBreathData", 0.1f, 0.033f);
    }

    void Update()
    {
        // 1. 更新 UI 显示
        if (tmpText != null) tmpText.text = _latestValueUpdate;

        // 2. 检查队列，把数据转移到“待上传列表”中
        while (dataQueue.TryDequeue(out EEGPoint point))
        {
            uploadBatch.Add(point);

            // 3. 攒够一波就上传
            if (uploadBatch.Count >= batchSize)
            {
                // 拷贝并清空，防止协程处理时列表被修改
                List<EEGPoint> toSend = new List<EEGPoint>(uploadBatch);
                uploadBatch.Clear();

                // 启动异步上传（不影响 VR 帧率）
                StartCoroutine(PostToTimescaleDB(toSend));
            }
        }
    }

    // 发送数据的协程
    IEnumerator PostToTimescaleDB(List<EEGPoint> batch)
    {
        // 将列表转为 JSON 字符串
        string json = JsonUtility.ToJson(new DataWrapper { items = batch });

        using (UnityWebRequest www = new UnityWebRequest("http://localhost:8000/upload_eeg", "POST"))
        {
            byte[] bodyRaw = System.Text.Encoding.UTF8.GetBytes(json);
            www.uploadHandler = new UploadHandlerRaw(bodyRaw);
            www.downloadHandler = new DownloadHandlerBuffer();
            www.SetRequestHeader("Content-Type", "application/json");

            yield return www.SendWebRequest();

            if (www.result != UnityWebRequest.Result.Success)
            {
                Debug.LogWarning("数据库写入失败: " + www.error);
            }
        }
    }

    [Serializable]
    public class DataWrapper { public List<EEGPoint> items; }


void updateBreathData()
    {  
        NXManager.GetInstance().updateBreathData();
    }

    // Step last, release all resources
    void OnDestroy()
    {
        CancelInvoke("updateBreathData");
        NXManager.GetInstance().deInitSDK();
    }

    public void BT_onConnectEvent(NXBTManager.BTConnectEvent status, byte[] message)
    {
        // TODO
        if (NXBTManager.BTConnectEvent.EVENT_BT_SCAN_STOP == status && mBluetoothDeviceAddress.Count > 0)
        {   
            //you can choose BT address
            NXBTManager.connectDevice(mBluetoothDeviceAddress.First());
        }
    }

    public void BT_onDeviceEvent(NXBTManager.BTDeviceEvent status, byte[] value)
    {
        // TODO
    }

    public void BT_onScanDevice(byte[] mac, byte[] name)
    {
        // TODO
        string macStr = BitConverter.ToString(mac);
        Debug.Log($"[Neuro] 发现设备: {macStr}"); // 看看有没有搜到设备
        // get device address,you call save address as list then choose the device that you want to connect.
        if (!mBluetoothDeviceAddress.Contains(mac))
        {
            mBluetoothDeviceAddress.Add(mac);
        }
    }

    public void BCI_onBandPowerEvent(double[] data)
    {
        // TODO        
    }

    public void BCI_onRawDataEvent(double[] data, int channel)
    {
        /*// TODO
        if (data != null && data.Length > 0)
        {
            string currentTimestamp = DateTime.Now.ToString("yyyy-MM-dd HH:mm:ss.fff");
            csvWriter.WriteLine($"{currentTimestamp},{channel},{data[0]}");
            Debug.Log($"[NeuroXess]channel {channel} eegdata: {data[0]}");
            tmpText.text = "channel"+ channel +"eegdata"+data[0];
        }*/
        if (data == null || data.Length == 0) return;

        // 获取系统当前时间作为辅助（防止硬件时间戳漂移）
        string sysTime = DateTime.UtcNow.ToString("yyyy-MM-dd HH:mm:ss.fff");

        // 遍历整个 data 数组（处理全部 250 个左右的采样点）
        for (int i = 0; i < data.Length; i++)
        {
            // 将每一个采样点都塞入队列
            dataQueue.Enqueue(new EEGPoint
            {
                ch = channel,
                val = data[i],
                sTime = sysTime
            });
        }

        // 仅在 UI 简单显示一下当前通道的最后一个值
        // 注意：这里不能直接写 tmpText.text，因为这是 SDK 线程
        
        _latestValueUpdate = $"Ch {channel}: {data[data.Length - 1]:F2}";
    }

    public void BCI_onPracticeEvent(NXBCIManager.BCIPracticeEvent status, double[] value)
    {
        switch(status)
        {
            case NXBCIManager.BCIPracticeEvent.EVENT_PRACTICE_CLOSED_EYES:
                //TODO  eyes closed
                break;            
            case NXBCIManager.BCIPracticeEvent.EVENT_PRACTICE_CURRENT_SCORE:
                //TODO value[0]:relax score, value[1]:focus score, value[2]:mindfulness score,value[3]:energy score
                break;            
        }
    }

    public void BRT_onBreathEvent(double a, int b, int c, int d)
    {
        // TODO
    }

    public void BRT_onRawDataEvent(double[] data, int channel)
    {
        // TODO
    }

    public void HRT_onHeartRate(int t)
    {
        // TODO
    }

    public void HRT_onRawDataEvent(double[] data, int channel)
    {
        // TODO
    }

    public void HRT_onHRVRMSSDEvent(double rmssd)
    {
        // TODO
    }

    public void AUTH_OnAuthorizeEvent(NXAuthManager.AuthorizeEvent events, byte[] msg)
    {
        // TODO
        Debug.Log($"[NeuroXess] state: {events}"); // 确认授权是否真的通过了
    }


    //
    // 绑定给“开始记录”按钮
   /* public void StartRecordingData()
    {
        if (isRecording) return;

        // 设置在 PICO 本地的保存路径 (Application.persistentDataPath 是 Android 推荐的安全存取路径)
        string timeStamp = DateTime.Now.ToString("yyyyMMdd_HHmmss");
        filePath = Path.Combine(Application.persistentDataPath, $"EEG_Data_{timeStamp}.csv");

        try
        {
            // 初始化写入流，并写入 CSV 表头
            csvWriter = new StreamWriter(filePath, true);
            csvWriter.WriteLine("Timestamp,Channel,DataValue");
            isRecording = true;
            Debug.Log($"[NeuroXess] start recording data，saved path: {filePath}");
            if (tmpText != null) tmpText.text = "state: start recording";
        }
        catch (Exception e)
        {
            Debug.LogError($"[NeuroXess] file create failed: {e.Message}");
        }
    }

    // 绑定给“暂停/停止记录”按钮
    public void StopRecordingData()
    {
        if (!isRecording) return;

        isRecording = false;

        if (csvWriter != null)
        {
            csvWriter.Flush(); // 确保缓存中的数据都写进磁盘
            csvWriter.Close(); // 关闭文件流
            csvWriter = null;
        }

        Debug.Log("[NeuroXess] stop recording, data saved。");
        if (tmpText != null) tmpText.text = "state: stop recording";
    }*/
    // 绑定给“开始记录”按钮
    public void StartRecordingHttpData()
    {
        if (isRecording) return;

        // 清理可能残留的旧数据，确保每次开始都是干净的起点
        while (dataQueue.TryDequeue(out _)) { }
        uploadBatch.Clear();

        isRecording = true;

        Debug.Log("[NeuroXess] 数据库写入已开启...");
        if (tmpText != null) tmpText.text = "state: start recording to DB";
    }

    // 绑定给“暂停/停止记录”按钮
    public void StopRecordingHttpData()
    {
        if (!isRecording) return;

        // 1. 立即关闭阀门，SDK 回调将不再往队列里塞新数据
        isRecording = false;

        // 2. 核心：强制发送残留数据 (类似之前的 csvWriter.Flush())
        FlushRemainingData();

        Debug.Log("[NeuroXess] 停止记录，尾部数据已冲刷至数据库。");
        if (tmpText != null) tmpText.text = "state: stop recording";
    }

    // 新增：强制冲刷缓冲区的辅助函数
    private void FlushRemainingData()
    {
        // 将队列中还没来得及转移的数据全拿出来
        while (dataQueue.TryDequeue(out EEGPoint point))
        {
            uploadBatch.Add(point);
        }

        // 如果上传列表中还有残留数据（即使不到 batchSize），直接强制发送
        if (uploadBatch.Count > 0)
        {
            List<EEGPoint> toSend = new List<EEGPoint>(uploadBatch);
            uploadBatch.Clear();
            StartCoroutine(PostToTimescaleDB(toSend));
        }
    }

}
