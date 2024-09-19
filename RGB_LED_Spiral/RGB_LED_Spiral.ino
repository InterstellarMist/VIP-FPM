// #include <ESP32-HUB75-MatrixPanel-I2S-DMA.h>
// #include <ESP32-VirtualMatrixPanel-I2S-DMA.h>

#include "ESP32-HUB75-MatrixPanel-I2S-DMA.h"

// Code from Waveshare
 
#define PANEL_WIDTH 64
#define PANEL_HEIGHT 64  	// Panel height of 64 will required PIN_E to be defined.
#define PANELS_NUMBER 1 	// Number of chained panels, if just a single panel, obviously set to 1
#define PIN_E 32

#define PANE_WIDTH PANEL_WIDTH * PANELS_NUMBER
#define PANE_HEIGHT PANEL_HEIGHT

int triggerPin = 21;

//MatrixPanel_I2S_DMA dma_display;
MatrixPanel_I2S_DMA *dma_display = nullptr;

void setup() {
  Serial.begin(115200);
  pinMode(triggerPin, OUTPUT);
  // Module configuration
  
  HUB75_I2S_CFG mxconfig;
  mxconfig.mx_height = PANEL_HEIGHT;      // we have 64 pix heigh panels
  mxconfig.chain_length = PANELS_NUMBER;    // we have 1 panel
  mxconfig.gpio.e = PIN_E;      // we MUST assign pin e to some free pin on a board to drive 64 pix height panels with 1/32 scan

  // Display Setup
  dma_display = new MatrixPanel_I2S_DMA(mxconfig);
  dma_display->begin();
  dma_display->setBrightness8(255); //0-255
  dma_display->clearScreen();

  Serial.println("Fill screen: RED");
  // RED
  // dma_display->fillScreenRGB888(255, 0, 0);
  // delay(100);

  // GREEN
  // dma_display->fillScreenRGB888(0, 255, 0);
  // delay(100);

  // // BLUE
  // dma_display->fillScreenRGB888(0, 0, 255);
  // delay(100);

  // // White
  // dma_display->fillScreenRGB888(255, 255, 255);
  // delay(100);

  // dma_display->clearScreen();

  // Draw Pixel
  // dma_display->drawPixelRGB888(31,31, 255, 0, 0);


  // Edge
  // dma_display->drawPixelRGB888(27+7,40, 255, 0, 0);

  // Corner
  // dma_display->drawPixelRGB888(28+4,35+4, 255, 0, 0);

  // Center
  // dma_display->drawPixelRGB888(28,35, 255, 255, 255);
  dma_display->drawPixelRGB888(28,35, 255, 0, 0);
  // Centering
  // centering(28,35,2,255,0,0);
  // centering(28,35,3,255,0,0);
  // centering(28,35,4,255,0,0);
}

void loop() {
  int interval = 425;
  // int interval = 225;
  // int arraysize = 11;
  int arraysize = 7;
  // spiral_animation(arraysize,28,35,interval, 255,0,0);
  // find_center();

  // RGB Capture
  // triggerLED(28,35,interval,255,0,0);
  // triggerLED(28,35,interval,0,255,0);
  // triggerLED(28,35,interval,0,0,255);
  // triggerLED(0,0,interval,0,0,255);
}

void centering(int x, int y,int gap,int r, int g, int b){
  dma_display->drawPixelRGB888(x,y+gap, r, g, b);
  dma_display->drawPixelRGB888(x,y-gap, r, g, b);
  dma_display->drawPixelRGB888(x+gap,y, r, g, b);
  dma_display->drawPixelRGB888(x-gap,y, r, g, b);
}

int find_center(){
  int x;
  int y;
  while (true) {
    if (Serial.available()){
      // receive coordinate "x,y"
      String input = Serial.readStringUntil('\n');
      int commaIndex = input.indexOf(',');
      x = input.substring(0,commaIndex).toInt();
      y = input.substring(commaIndex +1).toInt();
      Serial.print(x);
      Serial.print(',');
      Serial.println(y);
      triggerLED(y,x,500,255,0,0); 
    }
  }
}

void triggerLED(int x,int y,int interval, int r, int g, int b){
  dma_display->drawPixelRGB888(x, y, r, g, b);
  digitalWrite(triggerPin, HIGH);
  delayMicroseconds(100);
  digitalWrite(triggerPin, LOW);
  delay(interval);    
  dma_display->drawPixelRGB888(x, y, 0, 0, 0);
}

void spiral_animation(int arraysize, int centerx, int centery, int interval, int r, int g, int b) {
  int x = 0;
  int y = 0; // center position; x is col, y is row
  int dx = 0;
  int dy = 1; // initial displacement vector pointing downwards
     
  for (int i = 0; i < arraysize*arraysize; i++)
  {
    int m = -y+centery;
    int n = -x+centerx;
    Serial.print(i);
    Serial.print(' ');
    Serial.print(n);
    Serial.print(' ');
    Serial.println(m);
    triggerLED(n,m,interval,r,g,b);
    if (x == -y || (x < 0 && x == y) || (x > 0 && x == 1 + y))
    {
      int temp = dx;
      dx = dy;
      dy = -temp; // apply CCW rotation transformation
    }
    x += dx;
    y += dy;
  }
}

