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

public class Toggle_Controller : MonoBehaviour {

	string path1, path2, path3, path4, path5, path6;
	bool running = false;

	public Text run;

//	public Button LU, LD, RU, RD;

	string direction = "lu";

	void Start()
	{
		run = GameObject.Find("Running_Text").GetComponent<Text>();
//		LU = GameObject.Find ("Left Up").GetComponent<Button> ();
//		LD = GameObject.Find ("Left Down").GetComponent<Button> ();
//		RU = GameObject.Find ("Right Up").GetComponent<Button> ();
//		RD = GameObject.Find ("Right Down").GetComponent<Button> ();

		path1 = Application.persistentDataPath + "/" + direction + "_xaccl.txt".AppendTimeStamp();
		path2 = Application.persistentDataPath + "/" + direction + "_yaccl.txt".AppendTimeStamp();
		path3 = Application.persistentDataPath + "/" + direction + "_zaccl.txt".AppendTimeStamp();
		path4 = Application.persistentDataPath + "/" + direction + "_attitude.txt".AppendTimeStamp();
		path5 = Application.persistentDataPath + "/" + direction + "_rotrate.txt".AppendTimeStamp();
		path6 = Application.persistentDataPath + "/" + direction + "_userAccl.txt".AppendTimeStamp();
	}


	void Update()
	{
		float moveHorizontal = Input.acceleration.x; //GetAxis("Horizontal");
		float moveVertical = Input.acceleration.z; //GetAxis("Vertical");
		Input.gyro.enabled = true;
		//print("attitude: " + Input.gyro.attitude);
		//print("rot rate: " + Input.gyro.rotationRateUnbiased);
		//print("user: " + Input.gyro.userAcceleration);

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
		}
	}
}