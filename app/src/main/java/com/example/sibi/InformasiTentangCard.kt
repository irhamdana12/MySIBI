package com.example.sibi

import androidx.compose.foundation.Image
import androidx.compose.foundation.layout.*
import androidx.compose.foundation.rememberScrollState
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.foundation.verticalScroll
import androidx.compose.material3.*
import androidx.compose.runtime.Composable
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.draw.clip
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.res.painterResource
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.tooling.preview.Preview
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import androidx.navigation.NavHostController

@Composable
fun InformasiTentangCard(navController: NavHostController) {
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
                painter = painterResource(id = R.drawable.information),
                contentDescription = "Informasi Tentang SIBI Image",
                modifier = Modifier
                    .size(80.dp)
                    .clip(RoundedCornerShape(12.dp))
            )
            Spacer(modifier = Modifier.width(16.dp))
            Column {
                Text(
                    text = "Informasi Tentang SIBI",
                    style = MaterialTheme.typography.titleMedium
                )
                Spacer(modifier = Modifier.height(8.dp))
                Button(
                    onClick = { navController.navigate("informasiTentangSibi") },
                    colors = ButtonDefaults.buttonColors(containerColor = Color(0xFF00ADEF))
                ) {
                    Text("Kunjungi")
                }
            }
        }
    }
}

@Composable
fun InformasiTentangSibiScreen() {
    Column(
        modifier = Modifier
            .verticalScroll(rememberScrollState())
            .fillMaxSize()
            .padding(16.dp)

    ) {
        Spacer(modifier = Modifier.height(20.dp))
        Text(
            text = "Informasi Tentang SIBI",
            fontSize = 24.sp,
            fontWeight = FontWeight.Bold,
            modifier = Modifier.fillMaxWidth()
        )
        Spacer(modifier = Modifier.height(16.dp))
        // Card 1
        Card(
            modifier = Modifier
                .fillMaxWidth(),
            elevation = CardDefaults.cardElevation(8.dp)
        ) {
            Column(
                modifier = Modifier
                    .fillMaxWidth()
                    .padding(16.dp)
            ) {
                Text(
                    text = "Penjelasan Tentang SIBI",
                    style = MaterialTheme.typography.titleMedium, // Material 3
                    color = MaterialTheme.colorScheme.primary, // Material 3
                    fontWeight = FontWeight.Bold
                )
                Spacer(modifier = Modifier.height(8.dp))
                Text(
                    text = "Dalam aplikasi ini, menu \"Informasi tentang SIBI\" berfungsi untuk memberikan pengguna pengetahuan." + "yang komprehensif mengenai Sistem Isyarat Bahasa Indonesia (SIBI, " +
                            "SIBI adalah sistem bahasa isyarat resmi yang dikembangkan oleh pemerintah Indonesia untuk digunakan oleh penyandang disabilitas pendengaran." +
                            "  terutama dalam lingkungan pendidikan..",
                    style = MaterialTheme.typography.bodyMedium // Material 3
                )
            }
        }

        Spacer(modifier = Modifier.height(16.dp))

        // Card 2
        Card(
            modifier = Modifier
                .fillMaxWidth(),
            elevation = CardDefaults.cardElevation(8.dp)
        ) {
            Column(
                modifier = Modifier
                    .fillMaxWidth()
                    .padding(16.dp)
            ) {
                Text(
                    text = "Panduan Penggunaan SIBI",
                    style = MaterialTheme.typography.titleMedium, // Material 3
                    color = MaterialTheme.colorScheme.primary, // Material 3
                    fontWeight = FontWeight.Bold
                )
                Spacer(modifier = Modifier.height(8.dp))
                Text(
                    text = "Panduan penggunaan SIBI dalam aplikasi ini mencakup beberapa langkah dasar seperti mempelajari gerakan SIBI," +"" +
                            " memahami tata bahasa yang mengikuti struktur bahasa Indonesia, dan menerapkan SIBI dalam komunikasi sehari-hari.",
                    style = MaterialTheme.typography.bodyMedium // Material 3
                )
            }
        }

        Spacer(modifier = Modifier.height(16.dp))

        // Card 3
        Card(
            modifier = Modifier
                .fillMaxWidth(),
            elevation = CardDefaults.cardElevation(8.dp)
        ) {
            Column(
                modifier = Modifier
                    .fillMaxWidth()
                    .padding(16.dp)
            ) {
                Text(
                    text = "Informasi Perkembangan SIBI",
                    style = MaterialTheme.typography.titleMedium, // Material 3
                    color = MaterialTheme.colorScheme.primary, // Material 3
                    fontWeight = FontWeight.Bold
                )
                Spacer(modifier = Modifier.height(8.dp))
                Text(
                    text = "Informasi perkembangan SIBI disediakan untuk menunjukkan bagaimana sistem ini telah berkembang dan disesuaikan sejak pertama kali diresmikan,"
                            +" serta bagaimana teknologi modern, seperti aplikasi mobile, mendukung pembelajaran SIBI secara lebih interaktif.",
                    style = MaterialTheme.typography.bodyMedium // Material 3
                )
            }
        }
        Spacer(modifier = Modifier.height(16.dp))

        // Card 2
        Card(
            modifier = Modifier
                .fillMaxWidth(),
            elevation = CardDefaults.cardElevation(8.dp)
        ) {
            Column(
                modifier = Modifier
                    .fillMaxWidth()
                    .padding(16.dp)
            ) {
                Text(
                    text = "Sumber Daya dan Referensi",
                    style = MaterialTheme.typography.titleMedium, // Material 3
                    color = MaterialTheme.colorScheme.primary, // Material 3
                    fontWeight = FontWeight.Bold
                )
                Spacer(modifier = Modifier.height(8.dp))
                Text(
                    text = "Aplikasi menyediakan sumber daya dan referensi yang mencakup buku-buku tentang SIBI, aplikasi mobile untuk pembelajaran, video tutorial, kursus online, serta informasi mengenai organisasi dan komunitas yang dapat mendukung proses belajar pengguna.",
                    style = MaterialTheme.typography.bodyMedium // Material 3
                )
            }
        }
        Spacer(modifier = Modifier.height(16.dp))

        // Card 2
        Card(
            modifier = Modifier
                .fillMaxWidth(),
            elevation = CardDefaults.cardElevation(8.dp)
        ) {
            Column(
                modifier = Modifier
                    .fillMaxWidth()
                    .padding(16.dp)
            ) {
                Text(
                    text = "Keuntungan Menggunakan SIBI",
                    style = MaterialTheme.typography.titleMedium, // Material 3
                    color = MaterialTheme.colorScheme.primary, // Material 3
                    fontWeight = FontWeight.Bold
                )
                Spacer(modifier = Modifier.height(8.dp))
                Text(
                    text = "Keuntungan menggunakan SIBI juga diuraikan, termasuk bagaimana SIBI meningkatkan inklusi sosial," +
                            "  menyediakan standarisasi dalam pendidikan, memfasilitasi komunikasi yang efektif, dan diakui sebagai bagian dari bahasa resmi Indonesia.",
                    style = MaterialTheme.typography.bodyMedium // Material 3
                )
            }
        }
        Spacer(modifier = Modifier.height(16.dp))

        // Card 2
        Card(
            modifier = Modifier
                .fillMaxWidth(),
            elevation = CardDefaults.cardElevation(8.dp)
        ) {
            Column(
                modifier = Modifier
                    .fillMaxWidth()
                    .padding(16.dp)
            ) {
                Text(
                    text = "Pengunaan Dalam Pendidikan",
                    style = MaterialTheme.typography.titleMedium, // Material 3
                    color = MaterialTheme.colorScheme.primary, // Material 3
                    fontWeight = FontWeight.Bold
                )
                Spacer(modifier = Modifier.height(8.dp))
                Text(
                    text = "SIBI sering digunakan di Sekolah Luar Biasa(SLB) untuk memfasilitasi komunikasi antara guru dan siswa." +
                            " ini membantu siswa tunarungu dalam memahami pelajaran dan berinteraksi dengan guru serta teman-teman" +
                            " sekelas. penggunaan SIBI di sekolah juga bertujuan untuk memberikan pendidikan yang inklusif dan " +
                            "merata bagi semua siswa..",
                    style = MaterialTheme.typography.bodyMedium // Material 3
                )
            }
        }
    }
}

@Preview(showBackground = true)
@Composable
fun PreviewInformasiTentangSibiScreen() {
    InformasiTentangSibiScreen()
}
