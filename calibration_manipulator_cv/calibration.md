## Калибровка камеры на манипуляторе

1. Запустить драйвер:

	$ roslaunch youbot_driver_ros_interface youbot_driver.launch

2. Запустить ноду управления манипулятором:

	$ rosrun arm_manipulation test_manipulation

3. Выбрать condition - pos и ввести последовательноо следующие значения координат:
    
Сначала:

	Position (x, y, z): 0.3 0 0.5
	Angle (phi5, alpha): 0 0 

Затем:

	Position (x, y, z): 0.3 0 -0.05
	Angle (phi5, alpha): 0 3.1415

4. Уменьшая координату z (команда из предыдущего пункта) небольшими шагами (например, по два миллиметра 0.002) опустить гриппер до косания пола. 

5. Забить координаты из Position (x, y, z) в файл calibration_info.yaml, который находится в:

	$ roscd arm_manipulation/config

5. Подложить шаблон (черный квадрат на белом фоне) точно под гриппер так, чтобы квадрат был четко по центру гриппера, а его грани параллельны граням гриппера. Шаблон нужно закрепить, чтобы он не сдвинулся при выполнение дальнейших операций.

6. Задать координаты:

Приподымание гриппера от пола:

	Position (x, y, z): 0.3 0 -0.05
	Angle (phi5, alpha): 0 3.1415
	
Затем (перевод в положение для работы камеры):

	Position (x, y, z): 0.3 0 0.5
	Angle (phi5, alpha): 0 0

7. Измерить при помощи линейки/рулетки расстояние от пола до камеры (до грани камеры, параллельной полу).

8. Запустить камеру:

    $ roslaunch red_cvision usb_cam_youbot_0.launch 

9. Перейти в каталог:

    $ roscd red_cvision/scripts/servers/libs

10. Открыть файл `pymeasuring.py` и заменить в нем строку с переменной `DESIRED_CONTOURE_NAMES` на

    ```DESIRED_CONTOURE_NAMES = ['squar50']```

11. Запустить ноду:

    $ rosrun red_cvision talker.py -l "расстояние, измеренное в п.7 в [мм]"

12. Вызвать сервис, который должен вернуть объект 'squar50' и его координаты:

    $ rosservice call /get_list_objects "{}" 
    
Зпомнить-записать координаты из строки 'coordinates_center_frame'.

13. Запустить:

	$ roslaunch arm_manipulation start_camera_calibration.launch

ВАЖНО, чтобы существовал топик `/arm_1/joint_states` (его можно проверить введя команду rostopic list). Теперь запишем коодринаты предмета в узел при помощи ROS сервиса.

	$ rosservice call /object_position "angle: [0, 0, 0]
	position: [x, y, z]"

здесь координаты x и y получены с камеры, а z - это расстояние измеренное в п.7 в [м]. Если перейти во кладку с узлом start_camera_calibration, там будут постоянно печататься числа в скобках. Скопируйте эти числа и внесите в файл start_manipulation.launch. К нему можно попасть:

	$ roscd arm_manipulation/launch

Дальше вводим сами числа в строки:

	<param name="camera_offset_x" type="double" value="0.0589958" />
	<param name="camera_offset_y" type="double" value="0.0079992" />
	<param name="camera_offset_z" type="double" value="-0.0250016" />

Процедура окончена.
