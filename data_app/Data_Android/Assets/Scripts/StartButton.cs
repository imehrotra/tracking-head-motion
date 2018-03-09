using UnityEngine;
using UnityEngine.UI;
using System;
using System.Collections;
using System.IO;

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

	string path1, path2, path3, path4, path5, path6;
	bool running = false;

	public Text run;
	public Text counter;
	string direction = "lu";
//	int cnt = 0;
//	string prevdir = direction;

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
	}

	void NewPath() {
		path1 = Application.persistentDataPath + "/" + direction + "_xaccl.txt".AppendTimeStamp();
		path2 = Application.persistentDataPath + "/" + direction + "_yaccl.txt".AppendTimeStamp();
		path3 = Application.persistentDataPath + "/" + direction + "_zaccl.txt".AppendTimeStamp();
		path4 = Application.persistentDataPath + "/" + direction + "_attitude.txt".AppendTimeStamp();
		path5 = Application.persistentDataPath + "/" + direction + "_rotrate.txt".AppendTimeStamp();
		path6 = Application.persistentDataPath + "/" + direction + "_userAccl.txt".AppendTimeStamp();
	}

	void Update()
	{
		var toggleGroup = GameObject.Find("Canvas").GetComponent<ToggleGroup>();
//		prevdir = direction;
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
				Debug.Log (t.name);
//				if (prevdir != direction) {
//					NewPath ();
//				}
				break;
			}
		}

		float moveHorizontal = Input.acceleration.x;
		float moveVertical = Input.acceleration.z;
		Input.gyro.enabled = true;

		//		Debug.Log (running);
		if (running) {
			//Write some text to the txt file
			StreamWriter writer1 = new StreamWriter (path1, true);
			writer1.WriteLine (moveHorizontal);
			writer1.Close ();

			float yaccl = Input.acceleration.y;
			StreamWriter writer2 = new StreamWriter (path2, true);
			writer2.WriteLine (yaccl);
			writer2.Close ();

			StreamWriter writer3 = new StreamWriter (path3, true);
			writer3.WriteLine (moveVertical);
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
		}
	}

	public void TaskOnClick() {
		Debug.Log("You clicked the button");
		if (running) {
			running = false;
			run.text = "STOPPED";
		} else {
			running = true;
			run.text = "RUNNING";

			int cnt = Int32.Parse(counter.text);
			cnt++;
			counter.text = cnt.ToString ();

			NewPath ();
		}
	}
		
}