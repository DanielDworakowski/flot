Publishes sensor_msgs::Range messages at 10Hz to /sonarN topics. Specifically populates range.

REQUIRES running src/export_gpio_pins.sh to export pins otherwise need to run as root.
(limitation of wiringPiSetupSys)

    rostopic echo /sonar_0
    ---
    header:
      seq: 525
      stamp:
        secs: 1462946345
        nsecs: 463606605
      frame_id: ''
    radiation_type: 0
    field_of_view: 0.0
    min_range: 0.0
    max_range: 20.0
    range: 7.08620691299
    ---
