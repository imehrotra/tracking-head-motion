using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using UnityEngine.UI;

public class ResetButton : MonoBehaviour {

	public Text counter;

	void Start () {
		// initialize counter var to GameObject
		counter = GameObject.Find("Counter").GetComponent<Text>();

	}

	/* Function called when reset button clicked */
	public void OnClick() {
		Debug.Log("You clicked the reset button");
		counter.text = "0";
	}
}
