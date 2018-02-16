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
			DateTime.Now.ToString ("MMddHHmm"),
			Path.GetExtension (fileName)
		);
	}
}

public class PlayerController : MonoBehaviour
{

    public float speed;

    private Rigidbody rb;

    void Start()
    {
        rb = GetComponent<Rigidbody>();
    }

    void FixedUpdate()
    {
		float moveHorizontal = Input.acceleration.x; //GetAxis("Horizontal");
		float moveVertical = Input.acceleration.z; //GetAxis("Vertical");
		print("x-dir: " + moveHorizontal);
		print("z-dir: " + moveVertical);

        Vector3 movement = new Vector3(moveHorizontal, 0.0f, -moveVertical);

        rb.AddForce(movement * speed);

		string path1 = Application.persistentDataPath + "/" + "xaccl.txt".AppendTimeStamp();
		string path2 = Application.persistentDataPath + "/" + "yaccl.txt".AppendTimeStamp();
		string path3 = Application.persistentDataPath + "/" + "zaccl.txt".AppendTimeStamp();

		//Write some text to the test.txt file
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
    }
}