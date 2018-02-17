using UnityEngine;
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

public class PlayerController : MonoBehaviour
{

    public float speed;

    private Rigidbody rb;
	string path1, path2, path3, path4, path5, path6;

    void Start()
    {
        rb = GetComponent<Rigidbody>();

		path1 = Application.persistentDataPath + "/" + "xaccl.txt".AppendTimeStamp();
		path2 = Application.persistentDataPath + "/" + "yaccl.txt".AppendTimeStamp();
		path3 = Application.persistentDataPath + "/" + "zaccl.txt".AppendTimeStamp();
		path4 = Application.persistentDataPath + "/" + "attitude.txt".AppendTimeStamp();
		path5 = Application.persistentDataPath + "/" + "rotrate.txt".AppendTimeStamp();
		path6 = Application.persistentDataPath + "/" + "userAccl.txt".AppendTimeStamp();
    }

    void FixedUpdate()
    {
		float moveHorizontal = Input.acceleration.x; //GetAxis("Horizontal");
		float moveVertical = Input.acceleration.z; //GetAxis("Vertical");
		Input.gyro.enabled = true;
		//print("attitude: " + Input.gyro.attitude);
		//print("rot rate: " + Input.gyro.rotationRateUnbiased);
		//print("user: " + Input.gyro.userAcceleration);

        Vector3 movement = new Vector3(moveHorizontal, 0.0f, -moveVertical);

        rb.AddForce(movement * speed);

		//string path1 = Application.persistentDataPath + "/" + "xaccl.txt".AppendTimeStamp();
		//string path2 = Application.persistentDataPath + "/" + "yaccl.txt".AppendTimeStamp();
		//string path3 = Application.persistentDataPath + "/" + "zaccl.txt".AppendTimeStamp();
		//string path4 = Application.persistentDataPath + "/" + "attitude.txt".AppendTimeStamp();
		//string path5 = Application.persistentDataPath + "/" + "rotrate.txt".AppendTimeStamp();
		//string path6 = Application.persistentDataPath + "/" + "userAccl.txt".AppendTimeStamp();

		//Write some text to the txt file
		StreamWriter writer1 = new StreamWriter(path1, true);
		writer1.WriteLine(moveHorizontal);
		writer1.Close();

		float yaccl = Input.acceleration.y;
		StreamWriter writer2 = new StreamWriter(path2, true);
		writer2.WriteLine(yaccl);
		writer2.Close();

		StreamWriter writer3 = new StreamWriter(path3, true);
		writer3.WriteLine(moveVertical);
		writer3.Close();

		StreamWriter writer4 = new StreamWriter(path4, true);
		writer4.WriteLine(Input.gyro.attitude);
		writer4.Close();

		StreamWriter writer5 = new StreamWriter(path5, true);
		writer5.WriteLine(Input.gyro.rotationRateUnbiased);
		writer5.Close();

		StreamWriter writer6 = new StreamWriter(path6, true);
		writer6.WriteLine(Input.gyro.userAcceleration);
		writer6.Close();
    }
}