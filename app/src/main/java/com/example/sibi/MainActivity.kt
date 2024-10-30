package com.example.sibi

import android.os.Bundle
import androidx.activity.ComponentActivity
import androidx.activity.compose.setContent
import androidx.compose.foundation.background
import androidx.compose.foundation.layout.*
import androidx.compose.material3.*
import androidx.compose.runtime.*
import androidx.compose.ui.Modifier
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.tooling.preview.Preview
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import androidx.navigation.NavHostController
import androidx.navigation.compose.NavHost
import androidx.navigation.compose.composable
import androidx.navigation.compose.rememberNavController
import com.example.sibi.ui.theme.SIBITheme

class MainActivity : ComponentActivity() {
    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)

        setContent {
            var showDialog by remember { mutableStateOf(true) }

            SIBITheme {
                Surface(
                    modifier = Modifier.fillMaxSize(),
                    color = MaterialTheme.colorScheme.background
                ) {
                    val detectionHelper = ObjectDetectionHelper(LocalContext.current) // Initialize outside remember
                    SIBIScaffold(detectionHelper)

                    if (showDialog) {
                        AlertDialog(
                            onDismissRequest = { showDialog = false },
                            title = { Text("Informasi") },
                            text = { Text("Aplikasi ini hanya dapat menerjemahkan 15 kata") },
                            confirmButton = {
                                TextButton(onClick = { showDialog = false }) {
                                    Text("OK")
                                }
                            }
                        )
                    }
                }
            }
        }
    }
}

@Composable
fun SIBIScaffold(objectDetectionHelper: ObjectDetectionHelper) {
    val navController = rememberNavController()

    NavHost(navController = navController, startDestination = "home") {
        composable("home") {
            HomeScreen(navController)
        }
        composable("deteksiIsyarat") {
            CameraPreviewScreen()
        }
        composable("informasiTentangSibi") {
            InformasiTentangSibiScreen()
        }
        composable("daftarKata") {
            DaftarKataScreen(navController)
        }
        composable("resultScreen/{selectedWords}") { backStackEntry ->
            val selectedWords = backStackEntry.arguments?.getString("selectedWords")?.split(",") ?: emptyList()
            ResultScreen(selectedWords)
        }
    }
}

@Composable
fun HomeScreen(navController: NavHostController) {
    Column(
        modifier = Modifier
            .fillMaxSize()
            .background(Color(0xFFF0F0F0))
            .padding(16.dp)
    ) {
        Spacer(modifier = Modifier.height(16.dp))
        Text(
            text = "Selamat Datang",
            fontSize = 24.sp,
            fontWeight = FontWeight.Bold,
            modifier = Modifier.fillMaxWidth()
        )
        Spacer(modifier = Modifier.height(16.dp))
        DeteksiIsyaratCard(navController)
        Spacer(modifier = Modifier.height(16.dp))
        InformasiTentangCard(navController)
        Spacer(modifier = Modifier.height(16.dp))
        DaftarKataCard(navController)
        Spacer(modifier = Modifier.weight(1f))
    }
}

@Preview(showBackground = true)
@Composable
fun SIBIScaffoldPreview() {
    SIBITheme {
        val detectionHelper = ObjectDetectionHelper(LocalContext.current) // Initialize outside remember
        SIBIScaffold(detectionHelper)
    }
}