package com.example.sibi

import android.content.Intent
import android.os.Bundle
import androidx.activity.ComponentActivity
import androidx.compose.foundation.Image
import androidx.compose.foundation.layout.*
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.Surface
import androidx.compose.runtime.Composable
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.layout.ContentScale
import androidx.activity.compose.setContent
import androidx.compose.ui.res.painterResource
import androidx.compose.ui.unit.dp
import com.example.sibi.ui.theme.SIBITheme
import kotlinx.coroutines.*

class SplashActivity : ComponentActivity() {
    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)

        setContent {
            SIBITheme {

                Surface(modifier = Modifier.fillMaxSize(), color = MaterialTheme.colorScheme.background) {
                    SplashScreenContent()
                }
            }
        }

        CoroutineScope(Dispatchers.Main).launch {
            delay(2000L)  //
            startActivity(Intent(this@SplashActivity, MainActivity::class.java))
            finish()
        }
    }
}

@Composable
fun SplashScreenContent() {
    Box(
        contentAlignment = Alignment.Center,
        modifier = Modifier.fillMaxSize()
    ) {

        Image(
            painter = painterResource(id = R.drawable.ic_hand_open),
            contentDescription = "Splash Logo",
            contentScale = ContentScale.Fit,
            modifier = Modifier.size(200.dp)
        )
    }
}
