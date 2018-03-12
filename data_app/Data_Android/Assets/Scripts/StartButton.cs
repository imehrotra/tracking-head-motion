using UnityEngine;
using UnityEngine.UI;
using System;
using System.Collections;
using System.IO;
using System.Net.Sockets;
using System.Text;


public static class MyExtensions
{
	public static string AppendTimeStamp(this string fileName)
	{
		return string.Concat (
			Path.GetFileNameWithoutExtension (fileName),
			DateTime.Now.ToString ("MMddHHmmss"),
			Path.GetExtension (fileName)
		);
	}
}

public class StartButton : MonoBehaviour {

	string path1, path2, path3, path4, path5, path6, path7;
	bool running = false;

	public Text run;
	public Text counter;
	string direction = "lu";


	// network vars
	public String host = "localhost";
	public Int32 port = 50000;

	internal Boolean socket_ready = false;
	internal String input_buffer = "";
	TcpClient tcp_socket;
	NetworkStream net_stream;

	StreamWriter socket_writer;
	StreamReader socket_reader;

	int send_cnt = 0;

	void Start()
	{
		run = GameObject.Find("RunningText").GetComponent<Text>();
		counter = GameObject.Find("Counter").GetComponent<Text>();

		path1 = Application.persistentDataPath + "/" + direction + "_xaccl.txt".AppendTimeStamp();
		path2 = Application.persistentDataPath + "/" + direction + "_yaccl.txt".AppendTimeStamp();
		path3 = Application.persistentDataPath + "/" + direction + "_zaccl.txt".AppendTimeStamp();
		path4 = Application.persistentDataPath + "/" + direction + "_attitude.txt".AppendTimeStamp();
		path5 = Application.persistentDataPath + "/" + direction + "_rotrate.txt".AppendTimeStamp();
		path6 = Application.persistentDataPath + "/" + direction + "_userAccl.txt".AppendTimeStamp();
		path7 = Application.persistentDataPath + "/" + "send.txt".AppendTimeStamp();

	}

	void NewPath() {
		path1 = Application.persistentDataPath + "/" + direction + "_xaccl.txt".AppendTimeStamp();
		path2 = Application.persistentDataPath + "/" + direction + "_yaccl.txt".AppendTimeStamp();
		path3 = Application.persistentDataPath + "/" + direction + "_zaccl.txt".AppendTimeStamp();
		path4 = Application.persistentDataPath + "/" + direction + "_attitude.txt".AppendTimeStamp();
		path5 = Application.persistentDataPath + "/" + direction + "_rotrate.txt".AppendTimeStamp();
		path6 = Application.persistentDataPath + "/" + direction + "_userAccl.txt".AppendTimeStamp();
		path7 = Application.persistentDataPath + "/" + "send.txt";
	}


	void Update()
	{
		var toggleGroup = GameObject.Find("Canvas").GetComponent<ToggleGroup>();

		foreach (Toggle t in toggleGroup.ActiveToggles()) {
			if (t.isOn == true) {
				switch (t.name) {
				case("LeftUp"):
					direction = "lu";
					break;
				case("LeftDown"):
					direction = "ld";
					break;
				case("RightUp"):
					direction = "ru";
					break;
				case("RightDown"):
					direction = "rd";
					break;
				case("Forward"):
					direction = "fd";
					break;
				case("Back"):
					direction = "bk";
					break;
				case("Noisy"):
					direction = "ns";
					break;
				default:
					direction = "lu";
					break;
				}
//				Debug.Log (t.name);
				break;
			}
		}

		float xaccl = Input.acceleration.x;
		float yaccl = Input.acceleration.y;
		float zaccl = Input.acceleration.z;
		Input.gyro.enabled = true;



		string received_data = readSocket();
		if (received_data != "")
		{
			// Do something with the received data,
			// print it in the log for now
			Debug.Log(received_data);
		}

		// TODO put data in buffer
		input_buffer = Input.gyro.rotationRateUnbiased.y.ToString() + ",";
		input_buffer += Input.gyro.rotationRateUnbiased.z.ToString () + ",";
		input_buffer += Input.gyro.userAcceleration.z.ToString () + ",";
		input_buffer += xaccl.ToString() + ",";
		input_buffer += DateTime.Now.ToString("mmss") + ",";
		input_buffer += send_cnt.ToString () + ",";

//		input_buffer += "\n";

//		ASCIIEncoding ascii = new ASCIIEncoding();
//		Debug.Log("Byte Count:" + ascii.GetByteCount(input_buffer));

//		// Send the buffer, clean it
//		Debug.Log("Sending: " + input_buffer);
//		writeSocket(input_buffer);
//		input_buffer = "";


		// if running, write data to txt files
		if (running) {

			// Send the buffer, clean it
//			Debug.Log("Sending: " + input_buffer);
//			writeSocket(input_buffer);
//			send_cnt++;

			StreamWriter writer1 = new StreamWriter (path1, true);
			writer1.WriteLine (xaccl);
			writer1.Close ();

			StreamWriter writer2 = new StreamWriter (path2, true);
			writer2.WriteLine (yaccl);
			writer2.Close ();

			StreamWriter writer3 = new StreamWriter (path3, true);
			writer3.WriteLine (zaccl);
			writer3.Close ();

			StreamWriter writer4 = new StreamWriter (path4, true);
			writer4.WriteLine (Input.gyro.attitude);
			writer4.Close ();

			StreamWriter writer5 = new StreamWriter (path5, true);
			writer5.WriteLine (Input.gyro.rotationRateUnbiased);
			writer5.Close ();

			StreamWriter writer6 = new StreamWriter (path6, true);
			writer6.WriteLine (Input.gyro.userAcceleration);
			writer6.Close ();

			StreamWriter writer7 = new StreamWriter (path7, true);
			writer7.WriteLine(input_buffer);
			writer7.Close ();
		}

		input_buffer = "";
	}

	public void TaskOnClick() {
		Debug.Log("You clicked the button");
		if (running) {
			running = false;
			run.text = "STOPPED";

			string body = File.ReadAllText(Application.persistentDataPath + "/" + "send.txt");
			Debug.Log("Sending: input_buffer");
			writeSocket(body);
			send_cnt++;
			body = "";
//			input_buffer = "";
		} else {
			running = true;
			run.text = "RUNNING";

			int cnt = Int32.Parse(counter.text);
			cnt++;
			counter.text = cnt.ToString ();

			NewPath ();
		}
	}



	void Awake()
	{
		setupSocket();
	}

	void OnApplicationQuit()
	{
		closeSocket();
	}

	public void setupSocket()
	{
		try
		{
			tcp_socket = new TcpClient(host, port);

			net_stream = tcp_socket.GetStream();
			socket_writer = new StreamWriter(net_stream);
			socket_reader = new StreamReader(net_stream);

			socket_ready = true;
		}
		catch (Exception e)
		{
			// Something went wrong
			Debug.Log("Socket error: " + e);
		}
	}

	public void writeSocket(string line)
	{
		if (!socket_ready)
			return;

		line = line + "\r\n";
		socket_writer.Write(line);
		socket_writer.Flush();
	}

	public String readSocket()
	{
		if (!socket_ready)
			return "";

		if (net_stream.DataAvailable)
			return socket_reader.ReadLine();

		return "";
	}

	public void closeSocket()
	{
		if (!socket_ready)
			return;

		socket_writer.Close();
		socket_reader.Close();
		tcp_socket.Close();
		socket_ready = false;
	}


}