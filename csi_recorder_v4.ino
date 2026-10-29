#include <WiFi.h>
#include "esp_wifi.h"
#include <math.h>
#include <string.h>
#include "wifi_credentials.h"   // <-- credentials live here, NOT committed to git

#define MIN_FRAME_INTERVAL_MS 20
#define MAX_CSI_BYTES 256

int8_t   csi_copy[MAX_CSI_BYTES];
volatile uint16_t copied_length = 0;
volatile bool     csi_available = false;
portMUX_TYPE      csi_mux = portMUX_INITIALIZER_UNLOCKED;

unsigned long frame_number  = 0;
unsigned long last_print_ms = 0;

// ── CSI CALLBACK ─────────────────────────────────
void csi_callback(void* ctx, wifi_csi_info_t* info) {
  if (info == nullptr || info->buf == nullptr) return;
  uint16_t length = info->len;
  if (length < 4 || length > MAX_CSI_BYTES) return;

  portENTER_CRITICAL_ISR(&csi_mux);
  memcpy(csi_copy, info->buf, length);
  copied_length = length;
  csi_available = true;
  portEXIT_CRITICAL_ISR(&csi_mux);
}

// ── PRINT ONE CSI FRAME ──────────────────────────
void printCSIFrame(const int8_t* data, uint16_t length) {
  int sample_count = length / 2;
  if (sample_count <= 0) return;

  Serial.print(millis());
  Serial.print(",");
  Serial.print(frame_number++);
  Serial.print(",");
  Serial.print(sample_count);

  for (int i = 0; i < sample_count; i++) {
    float im = (float)data[2 * i];
    float re = (float)data[2 * i + 1];
    Serial.print(",");
    Serial.print(sqrtf(re * re + im * im), 3);
  }
  Serial.println();
}

// ── SETUP ────────────────────────────────────────
void setup() {
  Serial.begin(115200);
  delay(2000);

  Serial.println("# ESP32-S3 CSI Recorder v4");
  Serial.println("# Mode: Promiscuous (captures all nearby WiFi packets)");
  Serial.println("# Format: timestamp,frame_number,sample_count,amplitudes");

  WiFi.mode(WIFI_STA);
  WiFi.begin(WIFI_SSID, WIFI_PASSWORD);
  Serial.print("# Connecting to WiFi");

  while (WiFi.status() != WL_CONNECTED) {
    delay(400);
    Serial.print(".");
  }

  Serial.println();
  Serial.println("# WiFi connected");
  Serial.print("# IP: ");
  Serial.println(WiFi.localIP());
  Serial.print("# Channel: ");
  Serial.println(WiFi.channel());

  // ── CSI CONFIG ───────────────────────────────
  wifi_csi_config_t csi_config = {};
  csi_config.lltf_en           = true;
  csi_config.htltf_en          = true;
  csi_config.stbc_htltf2_en    = true;
  csi_config.ltf_merge_en      = true;
  csi_config.channel_filter_en = false;
  csi_config.manu_scale        = false;
  csi_config.shift             = false;

  esp_err_t result;

  result = esp_wifi_set_csi_config(&csi_config);
  if (result != ESP_OK) {
    Serial.print("# ERROR — CSI config: "); Serial.println(result); return;
  }

  result = esp_wifi_set_csi_rx_cb(csi_callback, nullptr);
  if (result != ESP_OK) {
    Serial.print("# ERROR — CSI callback: "); Serial.println(result); return;
  }

  result = esp_wifi_set_csi(true);
  if (result != ESP_OK) {
    Serial.print("# ERROR — CSI enable: "); Serial.println(result); return;
  }

  // ── PROMISCUOUS MODE ─────────────────────────
  esp_wifi_set_promiscuous(true);

  Serial.println("# Promiscuous mode: ON");
  Serial.println("# CSI enabled — capturing all nearby WiFi traffic");
  Serial.println("# DATA");
}

// ── LOOP ─────────────────────────────────────────
void loop() {
  int8_t   local_buffer[MAX_CSI_BYTES];
  uint16_t local_length = 0;
  bool     has_frame    = false;

  portENTER_CRITICAL(&csi_mux);
  if (csi_available) {
    local_length = copied_length;
    memcpy(local_buffer, csi_copy, local_length);
    csi_available = false;
    has_frame     = true;
  }
  portEXIT_CRITICAL(&csi_mux);

  if (has_frame) {
    unsigned long now = millis();
    if (now - last_print_ms >= MIN_FRAME_INTERVAL_MS) {
      printCSIFrame(local_buffer, local_length);
      last_print_ms = now;
    }
  }

  delay(1);
}
