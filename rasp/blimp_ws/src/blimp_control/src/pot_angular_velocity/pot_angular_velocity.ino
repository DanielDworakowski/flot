 
/* Analog Read to LED
 * ------------------ 
 *
 * turns on and off a light emitting diode(LED) connected to digital  
 * pin 13. The amount of time the LED will be on and off depends on
 * the value obtained by analogRead(). In the easiest case we connect
 * a potentiometer to analog pin 2.
 *
 * Created 1 December 2005
 * copyleft 2005 DojoDave <http://www.0j0.org>
 * http://arduino.berlios.de
 *
 */

float val = 0;       // variable to store the value coming from the sensor
float lastVal = 0;
int potPin = 2;
unsigned long lastTime = millis();
int T = 0;
float w = 0;

void setup() {
  Serial.begin(115200);
}

void loop() {
  T = lastTime - millis();
  lastTime = millis();
  val = 0.0;
  for(int i=0; i<30; i++)
  {
    delay(10);
    val += analogRead(potPin)/30.0; 
  }
  w = (lastVal - val)*(3.14159/600)*(1000/T);
  lastVal = val;
  Serial.print(w);
  Serial.println(",");
  delay(100);                  // stop the program for some time
}
 
