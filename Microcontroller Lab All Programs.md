# Microcontroller Lab: All Programs

Oct 29, 2026 · @Dewashish\_Astrowithin

## How to run

### 8051 programs (Experiments 1–7) in Keil µVision

1. Project → New µVision Project → name and save.
2. Select device: Atmel → AT89C51 (or the lab's 8051 chip). "Copy STARTUP.A51?" → No.
3. Right-click Target 1 → Options for Target → Target tab → Xtal = 11.0592. Output tab → tick Create HEX File (only if loading onto the kit).
4. File → New → type the code → save as `.asm` (assembly) or `.c` (C, Experiment 7).
5. Right-click Source Group 1 → Add Existing Files → select the file.
6. Build: F7 → check "0 Error(s)".
7. Debug: Ctrl+F5 → run with F5, or step line by line with F11.
8. Observe:
   - Registers window (left): A, R0–R7, PSW, SP, DPTR.
   - View → Memory Windows → Memory 1: type `D:40H` (RAM) or `C:0400H` (ROM).
   - Peripherals → I/O-Ports → Port 1 / Port 2: pin values.
   - Peripherals → Timer → Timer 0, and Peripherals → Interrupt (Experiment 7).
   - View → Analysis Windows → Logic Analyzer → Setup → add `P1.0` (square waves).
9. Stop debugging: Ctrl+F5.
10. Loading onto the 8051 kit (if asked): connect USB, note the COM port, open the lab's flash tool (SST EasyIAP / Flash Magic), choose the device and COM port, select the .hex file, download, then reset the kit.

### PIC programs (Experiments 8–12) in MPLAB X + PICLoader

1. MPLAB X: File → New Project → Microchip Embedded → Standalone Project.
2. Family: Advanced 8-bit MCUs (PIC18) → Device: PIC18F4550 → Tool: Simulator → Compiler: XC8 → name and location → Finish.
3. Right-click project → Properties → XC8 linker → Option category: Additional options → Codeoffset = 800 → Apply → OK.
4. Right-click Source Files → New → C Source File → type the code → save.
5. Right-click project → Clean and Build → check BUILD SUCCESSFUL. The .hex file is in `dist/default/production`.
6. Connect power and USB, switch the kit on. Device Manager → Ports (COM & LPT) → note the COM number (unplug/replug to confirm which one).
7. Open PICLoader (separate app) → Programs → Settings → select the COM port. Leave Config Bits and EEPROM unticked.
8. Press F3 → press RESET on the kit → press F4.
9. File → Open → the .hex file → Programs → Write Device (F6) → wait for "Write Complete".
10. Set the slide switches (table below) → press RESET on the kit → program runs.

| Device | Port / pins | SW21 | SW22 | SW23 | Extra |
| --- | --- | --- | --- | --- | --- |
| LEDs (Exp 8, 9) | RB0–RB7 | 1-2 | 2-3 | – | Keys SW18 = RA2, SW19 = RA3 |
| LCD (Exp 10) | RB0–RB7, RS = RA0, EN = RA1 | – | 1-2 | – | Adjust contrast pot P1 |
| 7-segment (Exp 11) | RB0–RB7, select RA0 / RA1 | 2-3 | 2-3 | – | Common anode: 0 = ON |
| Stepper (Exp 12) | RC0, RC1, RC2, RC5 | – | – | – | Plug motor into its connector |

If nothing works, check in order: slide switches → `ADCON1 = 0x0F` → Codeoffset = 800 → correct COM port → F3, RESET, F4 order → RESET after writing.

## Experiment 1: Addition of two numbers

```asm
ORG 0000H
MOV R1,#03H
MOV R7,#05H
MOV A,R1
ADD A,R7
MOV R5,A
END
```

## Experiment 2: PSW flags, arithmetic and logic

**Add two numbers**

```asm
ORG 0000H
MOV A,#0AH
ADD A,#14H
END
```

**PSW check: BFH + 1BH**

```asm
ORG 0000H
MOV A,#0BFH
ADD A,#1BH
END
```

**+7 + +18**

```asm
ORG 0000H
MOV A,#+7
MOV R4,#+18
ADD A,R4
END
```

**-5 + -2**

```asm
ORG 0000H
MOV A,#-5
MOV R4,#-2
ADD A,R4
END
```

**-128 + -2 (overflow)**

```asm
ORG 0000H
MOV A,#-128
MOV R4,#-2
ADD A,R4
END
```

**AND**

```asm
ORG 00H
MOV A,#06H
ANL A,#07H
END
```

**Register addition**

```asm
ORG 0000H
MOV R1,#03H
MOV R2,#05H
MOV A,R1
ADD A,R2
MOV R4,A
END
```

**OR**

```asm
ORG 00H
MOV A,#06H
ORL A,#07H
END
```

**EX-OR**

```asm
ORG 00H
MOV A,#06H
XRL A,#07H
END
```

**NOT**

```asm
ORG 00H
MOV A,#06H
CPL A
END
```

## Experiment 3: Addressing modes

**Send 55H to P1 and P2 using names**

```asm
ORG 0000H
MOV A,#55H
MOV P1,A
MOV P2,A
END
```

**Send 55H to P1 and P2 using addresses**

```asm
ORG 0000H
MOV A,#55H
MOV 90H,A
MOV 0A0H,A
END
```

**Copy 35H into 40H–45H: direct addressing**

```asm
ORG 0000H
MOV A,#35H
MOV 40H,A
MOV 41H,A
MOV 42H,A
MOV 43H,A
MOV 44H,A
MOV 45H,A
END
```

**Copy 35H into 40H–45H: register indirect, without loop**

```asm
ORG 0000H
MOV A,#35H
MOV R0,#40H
MOV @R0,A
INC R0
MOV @R0,A
INC R0
MOV @R0,A
INC R0
MOV @R0,A
INC R0
MOV @R0,A
INC R0
MOV @R0,A
END
```

**Copy 35H into 40H–45H: with loop**

```asm
ORG 0000H
MOV A,#35H
MOV R0,#40H
MOV R2,#06
AGAIN: MOV @R0,A
INC R0
DJNZ R2,AGAIN
END
```

## Experiment 4: Register banks, stack, repeated addition

**Multiply 25 × 10 by repeated addition**

```asm
ORG 00H
MOV A,#00H
MOV R0,#10
MOV R1,#25
BACK: ADD A,R1
DJNZ R0,BACK
END
```

**Sum of first ten natural numbers**

```asm
ORG 00H
MOV R0,#10
MOV A,#00
BACK: ADD A,R0
DJNZ R0,BACK
END
```

**Use bank 3 and load the same value in R0–R3**

```asm
ORG 00H
MOV A,#10
SETB RS1
SETB RS0
MOV R0,A
MOV R1,A
MOV R2,A
MOV R3,A
END
```

## Experiment 5: Loops and CALL

**Load 55H into A and complement it 700 times**

```asm
ORG 00H
MOV A,#55H
MOV R0,#10
BACK1: MOV R1,#70
BACK: CPL A
DJNZ R1,BACK
DJNZ R0,BACK1
END
```

**Subroutine that toggles an LED on P1.0**

```asm
ORG 00H
START: ACALL TOGGLE
SJMP START
TOGGLE: CPL P1.0
RET
END
```

**Toggle P1.0 without subroutine**

```asm
ORG 00H
START: CPL P1.0
SJMP START
END
```

**Toggle P1.0 with delay**

```asm
ORG 00H
START: ACALL TOGGLE
ACALL DELAY
SJMP START
TOGGLE: CPL P1.0
RET
DELAY: MOV R7,#0FFH
BACK1: MOV R6,#0FFH
BACK: DJNZ R6,BACK
DJNZ R7,BACK1
RET
END
```

**If R0 = FFH, move FFH to R5**

```asm
ORG 00H
MOV R0,#0FFH
MOV A,R0
CJNE A,#0FFH,NEXT
MOV B,R0
MOV R5,B
NEXT: NOP
END
```

## Experiment 6: I/O ports, bit manipulation, ROM to RAM

**Toggle all bits of Port 1 using 55H and AAH**

```asm
ORG 00H
START: MOV P1,#55H
ACALL DELAY
MOV P1,#0AAH
ACALL DELAY
SJMP START
DELAY: MOV R7,#0FFH
BACK1: MOV R6,#0FFH
BACK: DJNZ R6,BACK
DJNZ R7,BACK1
RET
END
```

**50% duty cycle square wave on P1.0**

```asm
ORG 00H
START: SETB P1.0
ACALL DELAY
CLR P1.0
ACALL DELAY
SJMP START
DELAY: MOV R7,#0FFH
BACK1: MOV R6,#0FFH
BACK: DJNZ R6,BACK
DJNZ R7,BACK1
RET
END
```

**Transfer "SIT" from code memory (0400H) to internal RAM (60H)**

```asm
ORG 00H
MOV A,#00H
MOV DPTR,#0400H
MOV R0,#60H
BACK: CLR A
MOVC A,@A+DPTR
MOV @R0,A
INC DPTR
INC R0
CJNE A,#00H,BACK
ORG 0400H
DB "SIT",0
END
```

## Experiment 7: 8051 timers (C)

**Delay using Timer 0, Mode 1**

```c
#include <reg51.h>
void t0delay(void);
void main()
{
    while(1)
    {
        P1 = 0x55;
        t0delay();
        P1 = 0xAA;
        t0delay();
    }
}
void t0delay(void)
{
    TMOD = 0x01;
    TL0 = 0x00;
    TH0 = 0x00;
    TR0 = 1;
    while(TF0 == 0);
    TR0 = 0;
    TF0 = 0;
}
```

**Square wave using Timer 0 interrupt, Mode 2**

```c
#include <reg51.h>
sbit wave = P1^0;
void Timer0_ISR(void) interrupt 1
{
    wave = ~wave;
}
void main()
{
    TMOD = 0x02;
    TH0 = 0xF5;
    TL0 = 0xF5;
    IE = 0x82;
    TR0 = 1;
    while(1);
}
```

**LED on P2.3 blinking with timer delay (1 s)**

```c
#include <reg51.h>
sbit led = P2^3;
void t0delay(void)
{
    TMOD = 0x01;
    TH0 = 0x4B;
    TL0 = 0xFD;
    TR0 = 1;
    while(TF0 == 0);
    TR0 = 0;
    TF0 = 0;
}
void main()
{
    unsigned char i;
    while(1)
    {
        led = 0;
        for(i = 0; i < 20; i++)
            t0delay();
        led = 1;
        for(i = 0; i < 20; i++)
            t0delay();
    }
}
```

## Experiment 8: LED interfacing (PIC)

```c
#include <pic18f4550.h>
void delay(void);
int main()
{
    TRISB = 0x00;
    while(1)
    {
        LATB = 0xFF;
        delay();
        LATB = 0x00;
        delay();
    }
}
void delay(void)
{
    int i;
    for(i = 0; i < 30000; i++)
    {
    }
}
```

## Experiment 9: Switch interfacing (PIC)

**Sample: control two LEDs with two switches**

```c
#include <pic18f4550.h>
void delay(unsigned int time);
void main(void)
{
    unsigned char a, b;
    ADCON1 = 0x0F;
    TRISB = 0x00;
    TRISAbits.TRISA2 = 1;
    TRISAbits.TRISA3 = 1;
    LATB = 0x00;
    while(1)
    {
        a = PORTAbits.RA2;
        b = PORTAbits.RA3;
        delay(10);
        LATBbits.LATB0 = ~a;
        LATBbits.LATB1 = ~b;
    }
}
void delay(unsigned int time)
{
    unsigned int i, j;
    for(i = 0; i < time; i++)
        for(j = 0; j < 5000; j++);
}
```

**Practice (a): if RA2 = 1 send FFH, else 00H**

```c
#include <pic18f4550.h>
void main(void)
{
    ADCON1 = 0x0F;
    TRISB = 0x00;
    TRISAbits.TRISA2 = 1;
    while(1)
    {
        if(PORTAbits.RA2 == 1)
            LATB = 0xFF;
        else
            LATB = 0x00;
    }
}
```

**Practice (b): if RA3 = 1 flash alternate LEDs with AAH**

```c
#include <pic18f4550.h>
void delay(unsigned int time);
void main(void)
{
    ADCON1 = 0x0F;
    TRISB = 0x00;
    TRISAbits.TRISA3 = 1;
    while(1)
    {
        if(PORTAbits.RA3 == 1)
        {
            LATB = 0xAA;
            delay(50);
            LATB = 0x00;
            delay(50);
        }
        else
            LATB = 0x00;
    }
}
void delay(unsigned int time)
{
    unsigned int i, j;
    for(i = 0; i < time; i++)
        for(j = 0; j < 5000; j++);
}
```

## Experiment 10: LCD interfacing (PIC)

**Sample: display "SIT E&TC"**

```c
#include <pic18f4550.h>
#define RS LATAbits.LATA0
#define EN LATAbits.LATA1
#define dataport LATB
void delay(unsigned int time);
void lcdcmd(unsigned char val);
void lcddata(unsigned char c);
void main(void)
{
    unsigned char i;
    const char msg[] = "SIT E&TC";
    ADCON1 = 0x0F;
    TRISA = 0x00;
    TRISB = 0x00;
    EN = 0;
    delay(100);
    lcdcmd(0x38);
    lcdcmd(0x0E);
    lcdcmd(0x01);
    delay(50);
    lcdcmd(0x06);
    lcdcmd(0x83);
    for(i = 0; msg[i] != '\0'; i++)
        lcddata(msg[i]);
    while(1);
}
void lcdcmd(unsigned char val)
{
    dataport = val;
    RS = 0;
    EN = 1;
    delay(1);
    EN = 0;
    delay(5);
}
void lcddata(unsigned char c)
{
    dataport = c;
    RS = 1;
    EN = 1;
    delay(1);
    EN = 0;
    delay(5);
}
void delay(unsigned int time)
{
    unsigned int i, j;
    for(i = 0; i < time; i++)
        for(j = 0; j < 500; j++);
}
```

**Practice: PRN on line 1, name on line 2**

```c
#include <pic18f4550.h>
#define RS LATAbits.LATA0
#define EN LATAbits.LATA1
#define dataport LATB
void delay(unsigned int time);
void lcdcmd(unsigned char val);
void lcddata(unsigned char c);
void lcdstring(const char *s);
void main(void)
{
    ADCON1 = 0x0F;
    TRISA = 0x00;
    TRISB = 0x00;
    EN = 0;
    delay(100);
    lcdcmd(0x38);
    lcdcmd(0x0C);
    lcdcmd(0x01);
    delay(50);
    lcdcmd(0x06);
    lcdcmd(0x82);
    lcdstring("24070123036");
    lcdcmd(0xC0);
    lcdstring("DEWASHISH L.");
    while(1);
}
void lcdstring(const char *s)
{
    while(*s)
        lcddata(*s++);
}
void lcdcmd(unsigned char val)
{
    dataport = val;
    RS = 0;
    EN = 1;
    delay(1);
    EN = 0;
    delay(5);
}
void lcddata(unsigned char c)
{
    dataport = c;
    RS = 1;
    EN = 1;
    delay(1);
    EN = 0;
    delay(5);
}
void delay(unsigned int time)
{
    unsigned int i, j;
    for(i = 0; i < time; i++)
        for(j = 0; j < 500; j++);
}
```

## Experiment 11: Seven-segment display (PIC)

**Sample: display 0 to 9**

```c
#include <pic18f4550.h>
void delay(unsigned int time);
void main(void)
{
    unsigned char i;
    const unsigned char c[10] = {0xC0, 0xF9, 0xA4, 0xB0, 0x99,
                                 0x92, 0x82, 0xF8, 0x80, 0x90};
    ADCON1 = 0x0F;
    TRISB = 0x00;
    TRISA = 0x00;
    LATA = 0x03;
    while(1)
    {
        for(i = 0; i < 10; i++)
        {
            LATB = c[i];
            delay(100);
        }
    }
}
void delay(unsigned int time)
{
    unsigned int i, j;
    for(i = 0; i < time; i++)
        for(j = 0; j < 5000; j++);
}
```

**Practice 1: display 12 (multiplexing)**

```c
#include <pic18f4550.h>
void delay(unsigned int time);
void main(void)
{
    ADCON1 = 0x0F;
    TRISB = 0x00;
    TRISA = 0x00;
    while(1)
    {
        LATB = 0xFF;
        LATA = 0x01;
        LATB = 0xF9;
        delay(2);
        LATB = 0xFF;
        LATA = 0x02;
        LATB = 0xA4;
        delay(2);
    }
}
void delay(unsigned int time)
{
    unsigned int i, j;
    for(i = 0; i < time; i++)
        for(j = 0; j < 5000; j++);
}
```

**Practice 2: display A to F**

```c
#include <pic18f4550.h>
void delay(unsigned int time);
void main(void)
{
    unsigned char i;
    const unsigned char alpha[6] = {0x88, 0x83, 0xC6, 0xA1, 0x86, 0x8E};
    ADCON1 = 0x0F;
    TRISB = 0x00;
    TRISA = 0x00;
    LATA = 0x03;
    while(1)
    {
        for(i = 0; i < 6; i++)
        {
            LATB = alpha[i];
            delay(100);
        }
    }
}
void delay(unsigned int time)
{
    unsigned int i, j;
    for(i = 0; i < time; i++)
        for(j = 0; j < 5000; j++);
}
```

**Extension: count 00 to 15**

```c
#include <pic18f4550.h>
const unsigned char seg[10] = {0xC0, 0xF9, 0xA4, 0xB0, 0x99,
                               0x92, 0x82, 0xF8, 0x80, 0x90};
void delay(unsigned int time);
void show(unsigned char n, unsigned int t);
void main(void)
{
    unsigned char n;
    ADCON1 = 0x0F;
    TRISB = 0x00;
    TRISA = 0x00;
    while(1)
    {
        for(n = 0; n <= 15; n++)
            show(n, 100);
    }
}
void show(unsigned char n, unsigned int t)
{
    unsigned int k;
    for(k = 0; k < t; k++)
    {
        LATB = 0xFF;
        LATA = 0x01;
        LATB = seg[n / 10];
        delay(2);
        LATB = 0xFF;
        LATA = 0x02;
        LATB = seg[n % 10];
        delay(2);
    }
}
void delay(unsigned int time)
{
    unsigned int i, j;
    for(i = 0; i < time; i++)
        for(j = 0; j < 5000; j++);
}
```

## Experiment 12: Stepper motor (PIC)

```c
#include <p18f4550.h>
void delay(unsigned int time)
{
    unsigned int i, j;
    for(i = 0; i < time; i++)
        for(j = 0; j < 5000; j++);
}
void main(void)
{
    TRISC = 0x00;
    LATC = 0xFF;
    while(1)
    {
        LATC = 0x01;
        delay(10);
        LATC = 0x02;
        delay(10);
        LATC = 0x04;
        delay(10);
        LATC = 0x20;
        delay(10);
    }
}
```
