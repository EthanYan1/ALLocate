#include <U8glib.h>

byte XstepPin = 54;
byte XdirectionPin = 55;
byte XenablePin = 38;

byte YstepPin = 60;
byte YdirectionPin = 61;
byte YenablePin = 56;

// byte ZstepPin = 46;
// byte ZdirectionPin = 48;
// byte ZenablePin = 62;

byte EistepPin = 26;
byte EidirectionPin = 28;
byte EienablePin = 24;

// byte EiistepPin = 36;
// byte EiidirectionPin = 34;
// byte EiienablePin = 30;

byte ledPin = 13;

int millisbetweenSteps = 1;

int X_iterations = 35;
int X_step_length = 50;

int Y_step_length = 100;

int total_iterations = 2;

bool fast_x_movement = false;

void setup() { 

  // setup stuff
  Serial.begin(115200);
  digitalWrite(ledPin, LOW);
    
  // delay(2000);

  pinMode(ledPin, OUTPUT);

  pinMode(XdirectionPin, OUTPUT);
  pinMode(XstepPin, OUTPUT);
  pinMode(XenablePin, OUTPUT);
 
  pinMode(YdirectionPin, OUTPUT);
  pinMode(YstepPin, OUTPUT);
  pinMode(YenablePin, OUTPUT);

  pinMode(EidirectionPin, OUTPUT);
  pinMode(EistepPin, OUTPUT);
  pinMode(EienablePin, OUTPUT);

  // scan the image

  for(int runs = 0; runs < total_iterations; runs++){
    // move X motor CW
    digitalWrite(XdirectionPin, HIGH);
    for(int i = 0; i < X_iterations; i++){
      for(int n = 0; n < X_step_length; n++) {
        digitalWrite(XstepPin, HIGH);
        digitalWrite(XstepPin, LOW);
        
        delay(millisbetweenSteps);
        
        digitalWrite(ledPin, !digitalRead(ledPin));
      }
      if(!fast_x_movement){
        delay(1000);
      }
    }
    
    delay(1000);

    // move Y motor CW
    digitalWrite(YdirectionPin, LOW);
    for(int n = 0; n < Y_step_length; n++) {
      digitalWrite(YstepPin, HIGH);
      digitalWrite(YstepPin, LOW);
      
      delay(millisbetweenSteps);
      
      digitalWrite(ledPin, !digitalRead(ledPin));
    }
    
    
    delay(1000);
    
    // move X motor CCW
    digitalWrite(XdirectionPin, LOW);
    for(int i = 0; i < X_iterations; i++){
      for(int n = 0; n < X_step_length; n++) {
        digitalWrite(XstepPin, HIGH);
        digitalWrite(XstepPin, LOW);
        
        delay(millisbetweenSteps);
        
        digitalWrite(ledPin, !digitalRead(ledPin));
      }
      if(!fast_x_movement){
        delay(1000);
      }
    }
    

    delay(1000);

    // move Y motor CW
    digitalWrite(YdirectionPin, LOW);
    for(int n = 0; n < Y_step_length; n++) {
      digitalWrite(YstepPin, HIGH);
      digitalWrite(YstepPin, LOW);
      
      delay(millisbetweenSteps);
      
      digitalWrite(ledPin, !digitalRead(ledPin));
    }
    
    delay(1000);
  }
}

void loop() { 
  // nothing here
}

  // move Y motor CCW
  // digitalWrite(YdirectionPin, LOW);
  // for(int n = 0; n < numberOfSteps; n++) {
  //   digitalWrite(YstepPin, HIGH);
  //   digitalWrite(YstepPin, LOW);
    
  //   delay(millisbetweenSteps);
    
  //   digitalWrite(ledPin, !digitalRead(ledPin));
  // }