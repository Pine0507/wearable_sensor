#include <chrono>
#include <cstdint>
#include <cstdio>
#include "mbed.h"
#include "ADS1115_def.h"
// #include "MPU9250_def.h"
#include "ICM_20948.h"
#include "SerialStream.h"
#include "vl53l1_api.h"
#include "vl53l1_core.h"
// #include "vl53l1_def.h"
#include "vl53l1_ll_device.h"
#include "vl53l1_platform.h"

// http://os.mbed.com/users/MultipleMonomials/code/SerialStream/

Timer timer_for_print_elaps_us;
#define ENABLE_PRINT_ELAPS_US 0

// mbedstudioデバッグ用 1 でプリントされる
#define DEBUG_PRINT 0

#if DEBUG_PRINT
#define ON_DEBUG(func) \
  {                    \
    func;              \
  }
#else
#define ON_DEBUG(func) \
  {                    \
  }
#endif

#if ENABLE_PRINT_ELAPS_US
#define print_elaps_us(text, func)                                              \
  {                                                                             \
    int s = timer_for_print_elaps_us.elapsed_time().count();                    \
    func;                                                                       \
    ONDEBUG(printf("%s: %d us, \r\n", text,                                     \
                   <int> timer_for_print_elaps_us.elapsed_time().count() - s);) \
  }
#else
#define print_elaps_us(text, func) \
  {                                \
    func;                          \
  }
#endif

#define INTERVAL_US (10000) // 10 ms

//各マルチプレクサに接続するIMUセンサ数
#define MUX1_IMU_NUM 16
#define MUX2_IMU_NUM 14
#define ALL_IMU_NUM (MUX1_IMU_NUM + MUX2_IMU_NUM)
ICM_20948_I2C imu[ALL_IMU_NUM];

//送信するデータの指定
#define DSIZE (12 * ALL_IMU_NUM + 4 + 2) //(加速度2bit*3)+(角速度2bit*3) *　センサ数　＋　
#define HSIZE 2
#define CSIZE 1
#define PSIZE (HSIZE + DSIZE + CSIZE)

char packet[PSIZE];
char *data = packet + HSIZE;
// マルチプレクサ切り替えアドレス
const char mux_id[8] = {0x01, 0x02, 0x04, 0x08, 0x10, 0x20, 0x40, 0x80};

Ticker ticker;

// 110, 300, 600, 1200, 2400, 4800, 9600, 14400, 19200, 38400, 57600, 115200,
// 230400, 460800, 921600 officially supports: 9600, 19200, 115200

#if DEBUG_PRINT
BufferedSerial usbSerial(USBTX, USBRX, 115200); // tx, rx, baud_rate (default:
#else
BufferedSerial usbSerial(USBTX, USBRX, 230400); // tx, rx, baud_rate (default:
                                                // MBED_CONF_PLATFORM_DEFAULT_SERIAL_BAUD_RATE = 9600)
#endif
// BufferedSerial usbSerial(USBTX, USBRX, 230400); // tx, rx, baud_rate
// (default: MBED_CONF_PLATFORM_DEFAULT_SERIAL_BAUD_RATE = 9600)
SerialStream<BufferedSerial> pc(usbSerial);
// UnbufferedSerial usbSerial(USBTX, USBRX, 115200); // tx, rx, baud_rate
// (default: MBED_CONF_PLATFORM_DEFAULT_SERIAL_BAUD_RATE = 9600)
// SerialStream<UnbufferedSerial> pc(usbSerial);
DigitalOut led[4] = {DigitalOut(LED1), DigitalOut(LED2), DigitalOut(LED3),
                     DigitalOut(LED4)};

bool output_flag = false;

char ch = 0;
char tmp_rx[14];
// (axyz + gxyz)*7 + ad1 + ad2 (2 bytes each)
// + distanec*4 (2 bytes each)
// + timestamp_us (uint32)
// + elapsed_time_in_loop (uint16)
// + distance_sensor_read_num (uint8)
// 1 101 000 1 : start single conv, AIN1 GND, +-6.144V, single shot mode
// 111 0 0 0 11: 860 SPS, traditional comparator, active low, nonlatching
// comparator, disable comparator and set alert/dry pin high
// 1 110 000 1 : start single conv, AIN1 GND, +-6.144V, single shot mode
// 111 0 0 0 11: 860 SPS, traditional comparator, active low, nonlatching
// comparator, disable comparator and set alert/dry pin high
// const char imu_read_msg[1] = {MPU9250_ACCEL_XOUT_H};
const char imu_read_msg[1] = {AGB0_REG_ACCEL_XOUT_H};

// #define MULTIPLEXER1 0xE0 // 1110(A2)(A1)(A0)(R/W)// 1110 1110

// See Figure. 9 TCA9548A Address
// |FIXED      |HARDSEL |  |
// | 1| 1| 1| 0|A2|A1|A0|RW|

// MULTIPLEXER1
// | 1| 1| 1| 0| 0| 0| 1| 0|
// マルチプレクサ増えたらここ変えるE0 E2 E4 E6 E8 EA EC EE
#define MULTIPLEXER1 0xE0
#define MULTIPLEXER2 0xE2
// | 1| 1| 1| 0| 0| 1| 1| 0|

// I2C i2c1(I2C_SDA1, I2C_SCL1);       // sda, scl
I2C i2c1(p9, p10); // sda, scl
// I2C i2c2(p28, p27); // sda, scl

//============================================
#define VL53L1_NUM 4

// MPU9250: 0xD0
// ADS1115: 0x90
// MUX: 0xEE, 0xE6
// Sparkfun MUX: 0xE0
// Queue<int, 4> q; // little bit slow perhaps due to thread safety.
// int seq_ix[4]{1,2,3,4};

//============================================

uint8_t read_sensors(void);
void sensor_loop(void);

//------------------------
int main()
{
  ON_DEBUG(printf("1 ::: started!\n"));
  timer_for_print_elaps_us.start();
  led[1] = 1;

  i2c1.frequency(400 * 1000);
  // i2c2.frequency(1000 * 1000); // 1MHz for VL53L1CB

  led[0] = 1;
  led[1] = 0;
  led[2] = 0;
  led[3] = 0;

  ICM_20948_fss_t myFSS; // This uses a "Full Scale Settings" structure that can
                         // contain values for all configurable sensors

  myFSS.a = gpm16; // (ICM_20948_ACCEL_CONFIG_FS_SEL_e)
                   // gpm2
                   // gpm4
                   // gpm8
                   // gpm16

  myFSS.g = dps2000; // (ICM_20948_GYRO_CONFIG_1_FS_SEL_e)
                     // dps250
                     // dps500
                     // dps1000
                     // dps2000
  // Set up Digital Low-Pass Filter configuration
  ICM_20948_dlpcfg_t myDLPcfg; // Similar to FSS, this uses a configuration
                               // structure for the desired sensors
  myDLPcfg.a =
      acc_d473bw_n499bw; // (ICM_20948_ACCEL_CONFIG_DLPCFG_e)
                         // acc_d246bw_n265bw      - means 3db bandwidth is 246
                         // hz and nyquist bandwidth is 265 hz
                         // acc_d111bw4_n136bw
                         // acc_d50bw4_n68bw8
                         // acc_d23bw9_n34bw4
                         // acc_d11bw5_n17bw
                         // acc_d5bw7_n8bw3        - means 3 db bandwidth is 5.7
                         // hz and nyquist bandwidth is 8.3 hz acc_d473bw_n499bw

  myDLPcfg.g = gyr_d361bw4_n376bw5; // (ICM_20948_GYRO_CONFIG_1_DLPCFG_e)
                                    // gyr_d196bw6_n229bw8
                                    // gyr_d151bw8_n187bw6
                                    // gyr_d119bw5_n154bw3
                                    // gyr_d51bw2_n73bw3
                                    // gyr_d23bw9_n35bw9
                                    // gyr_d11bw6_n17bw8
                                    // gyr_d5bw7_n8bw9
                                    // gyr_d361bw4_n376bw5
  // init imu
  {
    char tx[2];

    for (int i = 0; i < MUX1_IMU_NUM; i++)
    {
      ON_DEBUG(printf("[INFO] Start initiate imu[%d]\n", i);)
      bool ad0val = i % 2 == 0;
      char i2c_addr = ad0val ? ICM_20948_I2C_ADDR_AD0 : ICM_20948_I2C_ADDR_AD1;
      int mux_i = i / 2;
      int status = i2c1.write(MULTIPLEXER1, &mux_id[mux_i], 1);
      ON_DEBUG(printf("[INFO] Wrote %#x mux at status=%d\n", mux_id[mux_i], status);)

      wait_us(10);
      do
      {
        ON_DEBUG(printf("[INFO] begin imu addr=0x%#02x\n", i2c_addr);)
        imu[i].begin(i2c1, ad0val);
        ON_DEBUG(printf("[INFO] %s\n", imu[i].statusString());)
        if (imu[i].status != ICM_20948_Stat_Ok)
        {
          ON_DEBUG(printf("Trying again...\n");)
          wait_us(5000 * 1000);
        }
        else
        {
          break;
        }
      } while (1);

      // Here we are doing a SW reset to make sure the device starts in a known
      // state
      imu[i].swReset();
      if (imu[i].status != ICM_20948_Stat_Ok)
      {
        ON_DEBUG(printf("Software Reset returned: %s\n", imu[i].statusString());)
      }
      // wait_us(250 * 1000);
      wait_us(2500);

      // Now wake the sensor up
      imu[i].sleep(false);
      imu[i].lowPower(false);

      imu[i].setSampleMode((ICM_20948_Internal_Acc | ICM_20948_Internal_Gyr),
                           ICM_20948_Sample_Mode_Continuous);
      if (imu[i].status != ICM_20948_Stat_Ok)
      {
        ON_DEBUG(printf("setSampleMode returned: %s\n", imu[i].statusString());)
      }

      imu[i].setFullScale((ICM_20948_Internal_Acc | ICM_20948_Internal_Gyr),
                          myFSS);
      if (imu[i].status != ICM_20948_Stat_Ok)
      {
        ON_DEBUG(printf("setFullScale returned: %s\n", imu[i].statusString());)
      }

      imu[i].setDLPFcfg((ICM_20948_Internal_Acc | ICM_20948_Internal_Gyr),
                        myDLPcfg);
      if (imu[i].status != ICM_20948_Stat_Ok)
      {
        ON_DEBUG(printf("setDLPcfg returned: %s\n", imu[i].statusString());)
      }
      // Choose whether or not to use DLPF
      // Here we're also showing another way to access the status values, and
      // that it is OK to supply individual sensor masks to these functions
      ICM_20948_Status_e accDLPEnableStat =
          imu[i].enableDLPF(ICM_20948_Internal_Acc, false);
      ICM_20948_Status_e gyrDLPEnableStat =
          imu[i].enableDLPF(ICM_20948_Internal_Gyr, false);
      ON_DEBUG(printf("[INFO] Enable DLPF for Accelerometer returned: %s\n",
                      imu[i].statusString(accDLPEnableStat));)
      ON_DEBUG(printf("[INFO] Enable DLPF for Gyroscope returned: %s\n",
                      imu[i].statusString(gyrDLPEnableStat));)

      // Choose whether or not to start the magnetometer
      imu[i].startupMagnetometer();
      if (imu[i].status != ICM_20948_Stat_Ok)
      {
        ON_DEBUG(printf("startupMagnetometer returned: %s\n", imu[i].statusString());)
      }
    }

    const char close_port = 0x00;
    i2c1.write(MULTIPLEXER1, &close_port, 1);
    wait_us(10);
    // MUX2個目初期化

    for (int i = MUX1_IMU_NUM; i < (MUX1_IMU_NUM + MUX2_IMU_NUM); i++)
    {
      ON_DEBUG(printf("[INFO] Start initiate imu[%d]\n", i);)
      bool ad0val = i % 2 == 0;
      char i2c_addr = ad0val ? ICM_20948_I2C_ADDR_AD0 : ICM_20948_I2C_ADDR_AD1;
      int mux_i = (i - MUX1_IMU_NUM) / 2;
      int status = i2c1.write(MULTIPLEXER2, &mux_id[mux_i], 1);
      ON_DEBUG(printf("[INFO] Wrote %#x mux at status=%d\n", mux_id[mux_i], status);)

      wait_us(10);
      do
      {
        ON_DEBUG(printf("[INFO] begin imu addr=0x%#02x\n", i2c_addr);)
        imu[i].begin(i2c1, ad0val);
        ON_DEBUG(printf("[INFO] %s\n", imu[i].statusString());)
        if (imu[i].status != ICM_20948_Stat_Ok)
        {
          ON_DEBUG(printf("Trying again...\n");)
          wait_us(5000 * 1000);
        }
        else
        {
          break;
        }
      } while (1);

      // Here we are doing a SW reset to make sure the device starts in a known
      // state
      imu[i].swReset();
      if (imu[i].status != ICM_20948_Stat_Ok)
      {
        ON_DEBUG(printf("Software Reset returned: %s\n", imu[i].statusString());)
      }
      // wait_us(250 * 1000);
      wait_us(2500);

      // Now wake the sensor up
      imu[i].sleep(false);
      imu[i].lowPower(false);

      imu[i].setSampleMode((ICM_20948_Internal_Acc | ICM_20948_Internal_Gyr),
                           ICM_20948_Sample_Mode_Continuous);
      if (imu[i].status != ICM_20948_Stat_Ok)
      {
        ON_DEBUG(printf("setSampleMode returned: %s\n", imu[i].statusString());)
      }

      imu[i].setFullScale((ICM_20948_Internal_Acc | ICM_20948_Internal_Gyr),
                          myFSS);
      if (imu[i].status != ICM_20948_Stat_Ok)
      {
        ON_DEBUG(printf("setFullScale returned: %s\n", imu[i].statusString());)
      }

      imu[i].setDLPFcfg((ICM_20948_Internal_Acc | ICM_20948_Internal_Gyr),
                        myDLPcfg);
      if (imu[i].status != ICM_20948_Stat_Ok)
      {
        ON_DEBUG(printf("setDLPcfg returned: %s\n", imu[i].statusString());)
      }
      // Choose whether or not to use DLPF
      // Here we're also showing another way to access the status values, and
      // that it is OK to supply individual sensor masks to these functions
      ICM_20948_Status_e accDLPEnableStat =
          imu[i].enableDLPF(ICM_20948_Internal_Acc, false);
      ICM_20948_Status_e gyrDLPEnableStat =
          imu[i].enableDLPF(ICM_20948_Internal_Gyr, false);
      ON_DEBUG(printf("[INFO] Enable DLPF for Accelerometer returned: %s\n",
                      imu[i].statusString(accDLPEnableStat));)
      ON_DEBUG(printf("[INFO] Enable DLPF for Gyroscope returned: %s\n",
                      imu[i].statusString(gyrDLPEnableStat));)

      // Choose whether or not to start the magnetometer
      imu[i].startupMagnetometer();
      if (imu[i].status != ICM_20948_Stat_Ok)
      {
        ON_DEBUG(printf("startupMagnetometer returned: %s\n", imu[i].statusString());)
      }
    }
  }

  led[1] = 1;

  // ================ start read thread
  Thread thread_sensor_loop;
  thread_sensor_loop.start(sensor_loop);
  //    ticker.attach(read_sensors, 10ms);

  ON_DEBUG(printf("end init\n");)
  led[2] = 1;

  // sensor_loop();
  // ON_DEBUG(printf("888:::sensorloop\n");)

  // while (1);
  //    pc.set_blocking(false);
  while (1)
  {
    if (pc.readable() != 1)
    {
      output_flag = false;
    }
    else
    {
      pc.read(&ch, sizeof(ch));

      if (ch == 's')
      {
        output_flag = true;
        led[3] = 1;
      }
      else if (ch == 'e')
      {
        output_flag = false;
        led[3] = 0;
      }
    }
    wait_us(10);
  }
} // main

uint8_t NewDataReady = 0;
uint16_t _distance = 0;
uint8_t _i = 0;
int ix = 0, status = 0;

uint8_t read_sensors()
{
  int status;
  const char close_port = 0x00;

  i2c1.write(MULTIPLEXER2, &close_port, 1);
  wait_us(10);

  for (int imu_i = 0; imu_i < MUX1_IMU_NUM; imu_i++)
  {
    int mux_i = imu_i / 2;
    bool ad0val = imu_i % 2 == 0;
    char i2c_addr = ad0val ? ICM_20948_I2C_ADDR_AD0 : ICM_20948_I2C_ADDR_AD1;
    int status = i2c1.write(MULTIPLEXER1, &mux_id[mux_i], 1);

    if (!ad0val)
    {
      ON_DEBUG(pc.printf("[INFO] read sensors imu_i=%d\n", imu_i);)
      // int status = i2c1.write(MULTIPLEXER1, &mux_id[mux_i], 1);
      if (output_flag)
      {
        ON_DEBUG(pc.printf("[INFO] changed to mux[%d]\n", mux_i);)
      }
      wait_us(50);
    }

    if (imu[imu_i].dataReady())
    {
      ON_DEBUG(printf("[INFO] read imu addr=0x%#02x\n", i2c_addr);)
      // ICM_20948_AGMT_t agmt = imu[sensor_num].getAG(false);
      status = i2c1.write(i2c_addr << 1, imu_read_msg, 1);
      status += i2c1.read(i2c_addr << 1, tmp_rx, 12);

      ON_DEBUG(printf("[INFO] tmp_rx: ");)
      for (int axis_i = 0; axis_i < 12; axis_i++)
      {
        ON_DEBUG(printf("0x%#02x,", tmp_rx[axis_i]);)
      }
      ON_DEBUG(printf("\n");)

      memcpy(&data[12 * imu_i], &tmp_rx[0], 12);
      // if (output_flag) {
      //     int16_t x = ((tmp_rx[0] << 8) | (tmp_rx[1] & 0xFF));
      //     ON_DEBUG(printf("07 ::: imu[%d] status=%d x=%d\n", sensor_num, status, x);)
      // }
    }
    wait_us(50);
  }

  i2c1.write(MULTIPLEXER1, &close_port, 1);
  wait_us(10);

  for (int imu_i = MUX1_IMU_NUM; imu_i < (MUX1_IMU_NUM + MUX2_IMU_NUM); imu_i++)
  {
    int mux_i = (imu_i - MUX1_IMU_NUM) / 2;
    bool ad0val = imu_i % 2 == 0;
    char i2c_addr = ad0val ? ICM_20948_I2C_ADDR_AD0 : ICM_20948_I2C_ADDR_AD1;
    int status = i2c1.write(MULTIPLEXER2, &mux_id[mux_i], 1);

    if (!ad0val)
    {
      ON_DEBUG(pc.printf("[INFO] read sensors imu_i=%d\n", imu_i);)
      // int status = i2c1.write(MULTIPLEXER1, &mux_id[mux_i], 1);
      if (output_flag)
      {
        ON_DEBUG(pc.printf("[INFO] changed to mux[%d]\n", mux_i);)
      }
      wait_us(50);
    }

    if (imu[imu_i].dataReady())
    {
      ON_DEBUG(printf("[INFO] read imu addr=0x%#02x\n", i2c_addr);)
      // ICM_20948_AGMT_t agmt = imu[sensor_num].getAG(false);
      status = i2c1.write(i2c_addr << 1, imu_read_msg, 1);
      status += i2c1.read(i2c_addr << 1, tmp_rx, 12);

      ON_DEBUG(printf("[INFO] tmp_rx: ");)
      for (int axis_i = 0; axis_i < 12; axis_i++)
      {
        ON_DEBUG(printf("0x%#02x,", tmp_rx[axis_i]);)
      }
      ON_DEBUG(printf("\n");)

      memcpy(&data[12 * imu_i], &tmp_rx[0], 12);
      // if (output_flag) {
      //     int16_t x = ((tmp_rx[0] << 8) | (tmp_rx[1] & 0xFF));
      //     ON_DEBUG(printf("07 ::: imu[%d] status=%d x=%d\n", sensor_num, status, x);)
      // }
    }
    wait_us(50);
  }

  return _i; // return how many distance sensor was read.
}

void sensor_loop()
{
  int us_gap_sum = 0;
  int interval_dynamic_adjust_us = 0;
  uint32_t last_stat_time_us =
      timer_for_print_elaps_us.elapsed_time().count() - 10 * 1000;
  uint32_t loop_start_time_us = 0;
  uint16_t loop_elaps_us = 0, _max = 0, _min = 12345;
  int to_next_us = 0;
  uint8_t distance_sensor_read_num = 0;

  for (uint32_t _count = 1;; _count++)
  {
    loop_start_time_us = timer_for_print_elaps_us.elapsed_time().count();

    distance_sensor_read_num = read_sensors();

    us_gap_sum += (loop_start_time_us - last_stat_time_us - 10 * 1000);
    // us_gap_mean = (float)us_gap_sum / (float)_count;

#if DEBUG_PRINT
    // #if 1
    if (_count % 1 == 0)
    {
      for (int i = 0; i < HSIZE; i++)
        packet[i] = 0x00;
      loop_elaps_us =
          timer_for_print_elaps_us.elapsed_time().count() - loop_start_time_us;
      if (_max < loop_elaps_us)
        _max = loop_elaps_us;
      if (loop_elaps_us < _min)
        _min = loop_elaps_us;
      // pc.printf("Interval: %d, %d, %d, %d, %d, %d, ", _min, _max, loop_elaps_us,
      //           interval_dynamic_adjust_us,
      //           loop_start_time_us - last_stat_time_us, us_gap_sum);
    }
#endif

    if (output_flag)
    {
      loop_elaps_us =
          timer_for_print_elaps_us.elapsed_time().count() - loop_start_time_us;

      memcpy(&data[12 * ALL_IMU_NUM], &loop_start_time_us, 4);
      memcpy(&data[12 * ALL_IMU_NUM + 4], &loop_elaps_us, 2);

      packet[PSIZE - 1] = data[0];
      for (int i = 1; i < DSIZE; i++)
      {
        packet[PSIZE - 1] ^= data[i];
      }
      for (int imu_i = 0; imu_i < ALL_IMU_NUM; imu_i++)
      {
        ON_DEBUG(printf("[INFO] imu_i=%02d: ", imu_i);)
        for (int axis_i = 0; axis_i < 12; axis_i++)
        {
          ON_DEBUG(printf("0x%#02x,", data[imu_i * 12 + axis_i]);)
        }
        ON_DEBUG(printf("\n");)
      }

      pc.write(packet, sizeof(packet));
    }

    last_stat_time_us = loop_start_time_us;

    if (us_gap_sum > 0)
      interval_dynamic_adjust_us += 1;
    else
      interval_dynamic_adjust_us -= 1;

    // INTERVAL_US - (timer_for_print_elaps_us.elapsed_time().count() -
    // loop_start_time_us ) - us_gap_mean - interval_dynamic_adjust_us
    to_next_us = INTERVAL_US + loop_start_time_us - us_gap_sum -
                 interval_dynamic_adjust_us;
    to_next_us = to_next_us - timer_for_print_elaps_us.elapsed_time().count();
#if DEBUG_PRINT
    // #if 1
    if (_count % 1 == 0)
    {
      // pc.printf("%d\n", to_next_us);
    }
#endif
    if (to_next_us > 0)
    {
      // wait_us(to_next_us);  // wait_ns(to_next_us*1000);
      wait_us(50);
    }

    // std::chrono::microseconds(to_next_us));
  }
}

VL53L1_MultiRangingData_t MultiRangingData;
VL53L1_MultiRangingData_t *pMultiRangingData = &MultiRangingData;