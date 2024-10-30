@file:OptIn(ExperimentalLayoutApi::class)

package com.example.sibi

import androidx.compose.animation.animateColorAsState
import androidx.compose.foundation.BorderStroke
import androidx.compose.foundation.Image
import androidx.compose.foundation.background
import androidx.compose.foundation.layout.*
import androidx.compose.foundation.shape.RoundedCornerShape

import androidx.compose.material3.*
import androidx.compose.material3.AlertDialogDefaults.containerColor
import androidx.compose.runtime.*
import androidx.compose.runtime.Composable
import androidx.compose.runtime.remember
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.draw.clip
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.layout.ContentScale
import androidx.compose.ui.res.painterResource
import androidx.compose.ui.text.input.ImeAction
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import androidx.navigation.NavHostController
import com.google.accompanist.flowlayout.FlowRow
import androidx.compose.runtime.getValue
import androidx.compose.runtime.setValue
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.tooling.preview.Preview
import androidx.navigation.compose.NavHost
import androidx.navigation.compose.composable
import androidx.navigation.compose.rememberNavController
import androidx.preference.contains
import kotlin.text.contains

@Composable
fun DaftarKataCard(navController: NavHostController) {
    Card(
        modifier = Modifier
            .fillMaxWidth()
            .height(120.dp),
        shape = RoundedCornerShape(16.dp),
        colors = CardDefaults.cardColors(containerColor = Color.White)
    ) {
        Row(
            modifier = Modifier
                .fillMaxSize()
                .padding(16.dp),
            verticalAlignment = Alignment.CenterVertically
        ) {
            Image(
                painter = painterResource(id = R.drawable.kata),
                contentDescription = "Daftar Kata Image",
                modifier = Modifier
                    .size(80.dp)
                    .clip(RoundedCornerShape(12.dp)),
                contentScale = ContentScale.Crop
            )
            Spacer(modifier = Modifier.width(16.dp))
            Column {
                Text(
                    text = "Penerjemah Bahasa Indonesia ke Bahasa SIBI",
                    style = MaterialTheme.typography.titleMedium
                )
                Spacer(modifier = Modifier.height(8.dp))
                Button(
                    onClick = { navController.navigate("daftarKata") },
                    colors = ButtonDefaults.buttonColors(containerColor = Color(0xFF00ADEF))
                ) {
                    Text("Kunjungi")
                }
            }
        }
    }
}
@Composable
fun DaftarKataScreen(navController: NavHostController) {
    val selectedWords = remember { mutableStateListOf<KataItem>() }

    // Data for images and descriptions
    val kataList = listOf(
        KataItem(R.drawable.a, "a"),
        KataItem(R.drawable.i, "i"),
        KataItem(R.drawable.u, "u"),
        KataItem(R.drawable.e, "e"),
        KataItem(R.drawable.o, "o"),
        KataItem(R.drawable.baik, "baik"),
        KataItem(R.drawable.bangku, "bangku"),
        KataItem(R.drawable.bel, "bel"),
        KataItem(R.drawable.dia, "dia"),
        KataItem(R.drawable.meja, "meja"),
        KataItem(R.drawable.pramuka, "pramuka"),
        KataItem(R.drawable.sakit, "sakit"),
        KataItem(R.drawable.saya, "saya"),
        KataItem(R.drawable.teman, "teman"),
        KataItem(R.drawable.tugas, "tugas"),
    )

    // Function to toggle selection
    fun toggleSelection(kataItem: KataItem) {
        if (selectedWords.contains(kataItem)) {
            selectedWords.remove(kataItem)
        } else {
            selectedWords.add(kataItem)
        }
    }

    // Function to generate the result
    fun generate() {
        val selectedWordTitles = selectedWords.joinToString(",") { it.title }
        navController.navigate("resultScreen/$selectedWordTitles")
    }

    Column(
        modifier = Modifier
            .fillMaxSize()
            .padding(16.dp)
    ) {
        // Use FlowRow to arrange buttons
        FlowRow(
            mainAxisSpacing = 8.dp,
            crossAxisSpacing = 8.dp,
            modifier = Modifier.fillMaxWidth()
        ) {
            kataList.forEach { kataItem ->
                val isSelected = selectedWords.contains(kataItem)

                // Animate background color
                val backgroundColor by animateColorAsState(
                    targetValue = if (isSelected) Color.Blue else Color.LightGray
                )

                // Animate text color
                val textColor by animateColorAsState(
                    targetValue = if (isSelected) Color.White else Color.Black
                )
                // Button with animated background and text color
                OutlinedButton(
                    onClick =  { toggleSelection(kataItem) },
                    shape = RoundedCornerShape(50), // Rounded corners
                    border = BorderStroke(1.dp, Color.Gray), // Border color and thickness
                    colors = ButtonDefaults.outlinedButtonColors(
                        contentColor = if (selectedWords.contains(kataItem))  Color.White else Color(0xFF5D4C86),
                        containerColor = if (selectedWords.contains(kataItem)) Color(0xFF5D4C86) else Color.Unspecified,// Text color similar to the one in your image
                    ),
                    modifier = Modifier
                        .padding(2.dp)
                ) {
                    Text(text = kataItem.title)
                }

            }
        }

        Spacer(modifier = Modifier.height(16.dp))

        // Button to trigger the image generation
        Button(
            onClick = { generate() },
            colors = ButtonDefaults.buttonColors(
                containerColor = Color(0xFF00ADEF)
            ),
            modifier = Modifier.fillMaxWidth()
        ) {
            Text(text = "Generate")
        }
    }
}

// ResultScreen to display selected images
@Composable
fun ResultScreen(selectedWords: List<String>) {
    // Data for images and descriptions
    val kataList = listOf(
        KataItem(R.drawable.a, "a"),
        KataItem(R.drawable.i, "i"),
        KataItem(R.drawable.u, "u"),
        KataItem(R.drawable.e, "e"),
        KataItem(R.drawable.o, "o"),
        KataItem(R.drawable.baik, "baik"),
        KataItem(R.drawable.bangku, "bangku"),
        KataItem(R.drawable.bel, "bel"),
        KataItem(R.drawable.dia, "dia"),
        KataItem(R.drawable.meja, "meja"),
        KataItem(R.drawable.pramuka, "pramuka"),
        KataItem(R.drawable.sakit, "sakit"),
        KataItem(R.drawable.saya, "saya"),
        KataItem(R.drawable.teman, "teman"),
        KataItem(R.drawable.tugas, "tugas"),
    )

    val selectedItems = kataList.filter { it.title in selectedWords }

    Column(
        modifier = Modifier
            .fillMaxSize()
            .padding(16.dp)
    ) {
        selectedItems.forEach { item ->
            KataCard(item)
            Spacer(modifier = Modifier.height(16.dp))
        }
    }
}

@Composable
fun KataCard(kataItem: KataItem) {
    Card(
        modifier = Modifier
            .fillMaxWidth()
            .height(200.dp),
        shape = RoundedCornerShape(16.dp),
        colors = CardDefaults.cardColors(containerColor = Color.White)
    ) {
        Column(
            modifier = Modifier.fillMaxSize(),
            horizontalAlignment = Alignment.CenterHorizontally
        ) {
            Image(
                painter = painterResource(id = kataItem.imageRes),
                contentDescription = kataItem.title,
                modifier = Modifier
                    .width(160.dp)
                    .height(150.dp)
                    .clip(RoundedCornerShape(12.dp)),
                contentScale = ContentScale.Crop
            )
            Spacer(modifier = Modifier.height(8.dp))
            Text(
                text = kataItem.title,
                fontSize = 18.sp,
                modifier = Modifier.padding(8.dp)
            )
        }
    }
}

// Data class for kata
data class KataItem(val imageRes: Int, val title: String)

