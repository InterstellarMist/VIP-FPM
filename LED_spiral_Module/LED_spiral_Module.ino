/*
Spiral 7x7 using Led Control Library
by Nicholas Chai
2/17/2024
*/
#include <LedControl.h>

int DIN = 11;
int CS = 7;
int CLK = 13;
int triggerPin = 4;

LedControl lc=LedControl(DIN, CLK, CS, 0);

void setup()
{
  // Initialize LED
  lc.shutdown(0,false);
  lc.setIntensity(0,15);
  lc.setScanLimit(0,4); // Default: 8
  delay(100);
  lc.clearDisplay(0);

  Serial.begin(9600);
  pinMode(triggerPin, OUTPUT);

  // Activate LED
  // activate_all(2);
  // lc.setLed(0,6,6,true);
  lc.setLed(0,3,3,true);
  // lc.setLed(0,1,1,true);

  // Centering
  // centering(1);
  // centering(2);
}

void loop()
{
  int interval = 35;
  // linear_animation();
  //  spiral_animation(interval);
  // centering(100);
  // delay(1000);
}

void triggerLED(int x,int y, int interval){
  lc.setLed(0,x,y,true);
  digitalWrite(triggerPin, HIGH);
  delayMicroseconds(100);
  digitalWrite(triggerPin, LOW);
  delay(interval);    
  lc.setLed(0,x,y,false);
}

void centering(int gap){
  lc.setIntensity(0,15);
  lc.setLed(0,3-gap,3,true);
  lc.setLed(0,3+gap,3,true);
  lc.setLed(0,3,3+gap,true);
  lc.setLed(0,3,3-gap,true);
}

void spiral_animation(int interval) {
  int x = 0;
  int y = 0; // center position; x is col, y is row
  int dx = 0;
  int dy = 1; // initial displacement vector pointing downwards
     
  for (int i = 0; i < 49; i++)
  {
    int m = -x+3;
    int n = -y+3;
    Serial.print(i);
    Serial.print(' ');
    Serial.print(m);
    Serial.print(',');
    Serial.println(n);
    triggerLED(n,m,interval);
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

void linear_animation() {
   for(int j=0;j<8;j++){
     for(int i=0;i<8;i++){
        Serial.print(j);
        Serial.print(',');
        Serial.println(i);
        lc.setLed(0,j,i,true);
        digitalWrite(triggerPin, HIGH);
        delay(3);
        digitalWrite(triggerPin, LOW);
        lc.setLed(0,j,i,false);
     }
  }
}

void activate_all(int n) {
   for(int j=0;j<n;j++){
     for(int i=0;i<n;i++){
        lc.setLed(0,j,i,true);
     }
  }
}