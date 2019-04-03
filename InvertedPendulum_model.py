#!/usr/bin/python
""" inverted pendulum emulation

"""

import math
import pygame
import sys
if sys.platform == 'win32':
    sys.path.append("c:/Users/agoston.lorincz/PycharmProjects/NeuralNetwork")
else:
    sys.path.append("/run/user/1000/gvfs/smb-share:server=turrisnas,share=movies/scripts/phyton_scripts/NeuralNetwork")
import neural_network_sceleton as nn
import teach_neural_network as teach_nn
import numpy as np
import _pickle as pickle
import os.path


# key parameters
# mass of the pole
pole_mass = 0.2  # unit: kg
# total length of the pole (mass center is expected at length /2)
pole_length = 0.6  # unit: m
# initial position of the pole
pole_start_angle_deg = 180.0  # unit: degree (compared to vertical position)
# mass of the cart
cart_mass = 0.5  # unit: kg
# length of the cart
cart_length = 0.5  # length, height unit: m
# height of the cart
cart_height = 0.1  # length, height unit: m
# friction between the cart and the rail (rail is not modelled)
cart_friction = 0.1  # unit: N / (m / sec)
# initial position of the cart
cart_position = 0.0  # unit: m
# time constant of the actuator output
motor_tau = .1  # unit:[]
# maximum of motor force
motor_max_force = 9.99  # unit: N
# screen setup (width*3 x width)
width = 300  # unit: pixel
# select control method:
#   0: no control
#   1: dirac response
#   2: PID
#   3: neural network
#   4: replay control trace from file
control_method = 2
# dirac response force
dirac_force = 10  # unit: N
# desired cart position
pos_dirac = 0  # 0: linear interpolation between points; 1: steps for points
requested_position_v = [0, -0, 2, 2, 4, 4, -4]  # unit: m
requested_position_t = [0, 10, 20, 25, 35, 50, 55]  # unit: s
t_end = 70  # unit: s
# maximum position difference used for PD control
pos_err_max = 10
# cart position error to switch to near PD settings
cart_PD_near = 0.01  # unit: m
# cart PD Gains
Kp_cart = 80  # unit: []
Kd_cart = 30  # unit: []
# pole PD Gains
Kp_pole = 150  # unit: []
Kd_pole = 70
# save trace to file?
save_trace_to_file = 0  # 1: save new trace; 0: do not save new trace
# teach neural network before run
teach_neural_control = 0
teach_method_ga = 1


class Pendulum (object):
    mass = 0.0
    friction = 0.0
    length = 0.0
    position = 0.0
    angle = 0.0
    speed = 0.0
    asp = 0.0
    aa = 0.0
    acc = 0.0
    mass_moment = 0.0


class TraceInputOutput (object):
    def __init__(self):
        self.input0 = []
        self.input1 = []
        self.input2 = []
        self.input3 = []
        self.output0 = []

pygame.init()
clock = pygame.time.Clock()


# physical model of the inverted pendulum based on acceleration linkage (further debugging is needed)
def calculate_accelerations_of_the_system(f_ext: float, a_pole: Pendulum, a_cart: Pendulum):
    # N = m * pole_acc_x
    # pole_pos_x = cart_pos - l * sin(ang)
    # pole_spd_x = spd - l * asp * cos(ang)
    # pole_acc_x = acc + l * asp^2 * sin(ang) - l * ang_acc * cos(ang)
    # P = m * (pole_acc_y +g)
    # pole_pos_y =  l * cos(ang)
    # pole_spd_y = - l * asp  * sin(ang)
    # pole_acc_y = - l * asp^2 * cos(ang) - l * ang_acc * sin(ang)
    # equation3: acc = 1 / M (F - N - b * spd)
    # equation4: ang_acc = 1/I(N l cos(ang) + P l sin(ang))
    cos_angle = math.cos(a_pole.angle)
    sin_angle = math.sin(a_pole.angle)
    m = a_pole.mass  # unit kg
    acc = a_cart.acc  # unit m/sec^2
    l = a_pole.length / 2  # unit: m
    ang_asp_sqr = math.pow(a_pole.asp, 2)  # unit (rad/sec)^2
    ang_acc = a_pole.aa  # unit: rad/sec^2
    m_c = a_cart.mass  # unit: kg
    spd = a_cart.speed  # unit: m/s
    friction = a_cart.friction  # unit: N/m/s
    i_p = a_pole.mass_moment  # unit: kg*m^2
    int_t = t_d / 2

    # P = m * (acc_y_p + g) = m * (- l * asp^2 * cos(ang) - l * ang_acc * sin(ang) + g)
    pole_acc_y_1 = - l * ang_asp_sqr * cos_angle
    pole_acc_y_2 = - l * ang_acc * sin_angle
    pole_acc_y = pole_acc_y_1 + pole_acc_y_2
    p = m * (pole_acc_y + g)

    # N = m * acc_x_p = m * (acc + l * asp^2 * sin(ang) - l * ang_acc * cos(ang))
    pole_acc_x_1 = acc
    pole_acc_x_2 = + l * ang_asp_sqr * sin_angle
    pole_acc_x_3 = - l * ang_acc * cos_angle
    pole_acc_x = pole_acc_x_1 + pole_acc_x_2 + pole_acc_x_3
    n = m * pole_acc_x

    # acc =1 / M * (F - N - b * spd)
    a_acc = 1 / m_c * (f_ext - n - friction * spd)

    # ang_acc = 1 / I * (N * l * cos(ang) + P * l * sin(ang))
    a_ang_acc = 1 / i_p * (n * l * cos_angle + p * l * sin_angle)

    # the pole acceleration
    a_pole.aa = a_ang_acc
    # calculate pole speed
    a_pole.asp = a_pole.asp + (a_pole.aa * int_t)
    # calculate pole current angle
    a_pole.angle = a_pole.angle + (a_pole.asp * int_t)
    # reduce the pole angle into -pi..pi
    if math.pi < a_pole.angle:
        a_pole.angle = a_pole.angle - (math.pi * 2)
    if a_pole.angle < -math.pi:
        a_pole.angle = a_pole.angle + (math.pi * 2)
    # calculate pole current angle in degree
    # pole_angle_deg = a_pole.angle / math.pi * 180
    # define cart acceleration
    a_cart.acc = a_acc
    # calculate cart current speed
    a_cart.speed = a_cart.speed + (a_cart.acc * int_t)
    # calculate cart current position
    a_cart.position = a_cart.position + (a_cart.speed * int_t)
    # calculate new pole position
    a_pole.x = cart.position - l * math.sin(a_pole.angle)
    a_pole.y = l * math.cos(a_pole.angle)

    # print('time', '{:1.4f}'.format(t),
    #       'ang_acc ', '{:15.1f}'.format(a_pole.aa),
    #       'ang_asp', '{:15.1f}'.format(a_pole.asp),
    #       'angle_deg', '{:15.1f}'.format(pole_angle_deg),
    #       'p', '{:15.1f}'.format(p),
    #       'n', '{:15.1f}'.format(n))

    return {'pole': a_pole, 'cart': a_cart}


# physical model of the inverted pendulum using force linkage
# see: https://github.com/Nikkhil16/Inverted_Pendulum/blob/master/inverted_pendulum.py
def calculate_accelerations_of_the_system_2(f_ext: float, a_pole: Pendulum, a_cart: Pendulum):
    cos_angle = math.cos(a_pole.angle)
    sin_angle = math.sin(a_pole.angle)
    m = a_pole.mass  # unit kg
    # acc = a_cart.acc  # unit m/sec^2
    l = a_pole.length / 2  # unit: m
    ang_asp_sqr = math.pow(a_pole.asp, 2)  # unit (rad/sec)^2
    # ang_acc = a_pole.aa  # unit: rad/sec^2
    m_c = a_cart.mass  # unit: kg
    spd = a_cart.speed  # unit: m/s
    friction = a_cart.friction  # unit: N/m/s
    # i_p = a_pole.mass_moment  # unit: kg*m^2
    int_t = t_d / 2

    # ang_acc =(((M + m) * g * sinang) + (F * cosang) - (m * (asp^2) * l * sinang * cosang)) / (l * (M + (m* sinang^2)))
    a_ang_acc = (((m_c + m) * g * sin_angle) + ((f_ext - spd * friction) * cos_angle) -
                 (m * ang_asp_sqr * l * sin_angle * cos_angle)) / \
                (l * (m_c + (m * math.pow(sin_angle, 2))))
    # acc !111111111111111111111111111111    QQQQQ ((m * g * sinang * cosang) - (m * l * sinang * asp^2) + F) / (M + (m * sinang^2))
    a_acc = ((m * g * sin_angle * cos_angle) - (m * l * sin_angle * ang_asp_sqr) + (f_ext - spd * friction)) / \
            (m_c + (m * math.pow(sin_angle, 2)))

    # the pole acceleration
    a_pole.aa = a_ang_acc
    # calculate pole speed
    a_pole.asp = a_pole.asp + (a_pole.aa * int_t)
    # calculate pole current angle
    a_pole.angle = a_pole.angle + (a_pole.asp * int_t)
    # reduce the pole angle into -pi..pi
    if math.pi < a_pole.angle:
        a_pole.angle = a_pole.angle - (math.pi * 2)
    if a_pole.angle < -math.pi:
        a_pole.angle = a_pole.angle + (math.pi * 2)
    # calculate pole current angle in degree
    # pole_angle_deg = a_pole.angle / math.pi * 180
    # define cart acceleration
    a_cart.acc = a_acc
    # calculate cart current speed
    a_cart.speed = a_cart.speed + (a_cart.acc * int_t)
    # calculate cart current position
    a_cart.position = a_cart.position + (a_cart.speed * int_t)
    # calculate new pole position
    a_pole.x = cart.position - l * math.sin(a_pole.angle)
    a_pole.y = l * math.cos(a_pole.angle)

    return {'pole': a_pole, 'cart': a_cart}


def calculate_motor_force(u: float):
    motor.dy = - 1 / motor.tau * motor.y + u / motor.tau
    y = motor.y + motor.dy * t_d
    return y


# class to store solar objects to draw
class ARectangle(pygame.sprite.Sprite):
    def __init__(self, color, size1, size2, angle):
        # create a sprite for each object
        pygame.sprite.Sprite.__init__(self)
        self.maxsize_s = size2 * 2
        if size2 < size1:
            self.maxsize_s = size1 * 2
        self.maxsize = int(self.maxsize_s)
        self.image = pygame.Surface((self.maxsize, self.maxsize))
        self.image.fill((128, 128, 128))
        self.Rect = (self.maxsize / 2 - size1, self.maxsize / 2 - size2, size1 * 2, size2 * 2)
        pygame.draw.rect(self.image, color, self.Rect, 0)
        self.rect = self.image.get_rect()
        # make background transparent
        self.image.set_colorkey((128, 128, 128))
        self.image_original = self.image
        x, y = self.rect.center
        alpha = float(angle) * 180 / math.pi
        self.image = pygame.transform.rotate(self.image_original, alpha)
        self.rect = self.image.get_rect()
        self.rect.center = (x, y)

    # update angle (delta!)
    def rotate_angle(self, alpha):
        alpha = float(alpha) * 180 / math.pi
        x, y = self.rect.center
        self.image = pygame.transform.rotate(self.image_original, alpha)
        self.rect = self.image.get_rect()
        self.rect.center = (x, y)

    # update position
    def set_new_position(self, scr_x, scr_y):
        self.rect.center = (scr_x, scr_y)


def control_pendulum(a_pole: Pendulum, a_cart: Pendulum):
    pos_err = 0
    speed_ref = 0
    ang_mod = 0
    # if the requested control method is no control
    if control_method == 0:
        force = 0
    # if the requested force is a little push at the beginning
    elif control_method == 1:
        if t == 0:
            force = dirac_force
        else:
            force = 0
    # if the requested control method is PD control
    elif control_method == 2:
        # define the error values
        pos_err = requested_position - a_cart.position
        # maximize the position error
        pos_err = max(min(pos_err, pos_err_max), - pos_err_max)
        # angle speed error
        asp_err = 0 - a_pole.asp
        # initialize the cart control output
        force_c = 0

        # position control of the cart
        if abs(a_pole.angle) < math.pi / 8:
            # define speed reference signal (with ramp, modified by the speed of the request signal)
            speed_ref = math.copysign(min(0.13 * abs(pos_err), abs(a_cart.speed) + 0.1), pos_err) * 6 + requested_speed
            # calculate speed error
            spd_err = speed_ref - a_cart.speed
            # P control for speed
            force_c = - spd_err * Kd_cart
            # to achieve more robust control support the cart movement with angle modification
            ang_mod = - spd_err * 0.02
            if abs(spd_err) < 0.02:
                ang_mod = 0
            if abs(pos_err) < cart_PD_near:
                ang_mod = 0
                force_c += a_cart.speed * Kd_cart * 4

        # angle control of the pole
        # if close to the requested upright position (near linear model)
        if abs(a_pole.angle) < math.pi / 4:
            # calculate angle error
            ang_err = ang_mod - a_pole.angle
            # PD output for pole
            force_p = ang_err * Kp_pole
            force_p += asp_err * Kd_pole
        # if the pole is far from the requested position, only speed control to get it to near position
        elif abs(a_pole.angle) > math.pi * 3 / 4:
            force_c = 0
            force_p = - math.copysign(max(abs(a_pole.asp) * 1.0, 0.1), a_pole.asp)
        # not close but also not far (around vertical position)
        else:
            force_c = 0
            force_p = 0
        # summarize cart and pole control outputs
        force = force_p + force_c
    # if neural network control is requested
    elif control_method == 3:
        # inputs:
        #   0: cart position
        #   1: pole angle
        #   2: cart requested position
        #   3: small angle error indicator
        # output:
        #   0: requested external force
        if abs(a_pole.angle) < 45 / 180 * math.pi:
            small_angle = 1
        else:
            small_angle = 0
        an_input_vector = np.array([a_cart.position, a_pole.angle, requested_position, small_angle])
        force = control_network.calculate_output(an_input_vector)
    # if control method is saved data
    elif control_method == 4:
        force = saved_trace.output0[0]
        saved_trace.output0.pop(0)
    # if control method selection is not defined
    else:
        force = 0

    print('time', '{:3.4f}'.format(t),
          '   ang_acc ', '{:6.1f}'.format(a_pole.aa),
          '   ang_asp', '{:5.1f}'.format(a_pole.asp),
          '   angle_deg', '{:6.1f}'.format(a_pole.angle * 180 / math.pi),
          '   ang_mod', '{:5.1f}'.format(ang_mod * 180 / math.pi),
          '   position ', '{:7.4f}'.format(a_cart.position),
          '   pos_err ', '{:7.4f}'.format(pos_err),
          '   speed', '{:7.4f}'.format(a_cart.speed),
          '   spd_ref', '{:7.4f}'.format(speed_ref),
          '   ext_force', '{:8.4f}'.format(external_force),
          '   req_pos', '{:7.4f}'.format(requested_position))

    # provide output limited by motor.max_force
    return max(min(force, motor.max_force), -motor.max_force)


# some color definition
black = (0, 0, 0)  # Fore- and background colors
white = (255, 255, 255)
red = (255, 0, 0)
blue = (0, 0, 255)
green = (0, 255, 0)
yellow = (255, 255, 0)
purple = (255, 0, 255)
# create display
screen = pygame.display.set_mode((width * 3, width), pygame.HWACCEL)
# create an other surface, so the the orbits can separately updated / drawn to the stars/planets/moons
background = pygame.Surface(screen.get_size())
# set background color
background.fill(black)
# at first copy the complete background to the display
screen.blit(background, (0, 0))
# create a sprites group so later these can be added to the display, and moved together
allSprites = pygame.sprite.Group()
# define 0 positions of the screen
screen_x_null = width * 3 / 2
screen_y_null = width * 0.7
screen_scale = 100

# define base parameters of the pole
pole = Pendulum()
pole.mass = pole_mass  # unit: kg
pole.length = pole_length  # unit: m
pole.angle = pole_start_angle_deg / 180 * math.pi  # unit: rad
pole.asp = 0.0  # unit: rad/sec
pole.aa = 0.0  # unit: rad/sec^2
pole.mass_moment = 1 / 3 * pole.mass * math.pow(pole.length / 2, 2)  # unit: kg*m^2
pole.rect = ARectangle(red, 1, pole.length * screen_scale / 2, pole.angle)

# define base parameters of the cart
cart = Pendulum()
cart.mass = cart_mass  # unit: kg
cart.length = cart_length  # length, height unit: m
cart.height = cart_height  # length, height unit: m
cart.position = cart_position  # unit: m
cart.speed = 0.0  # unit: m/s
cart.acc = 0.0  # unit: m/sec^2
cart.friction = cart_friction  # unit: N/m/sec
cart.rect = ARectangle(blue, cart.length * screen_scale / 2, cart.height * screen_scale / 2, 0)

# place cart on the screen
cart.x_scr = screen_x_null  # unit: pixel
cart.y_scr = screen_y_null  # unit: pixel
cart.rect.set_new_position(cart.x_scr, cart.y_scr)
# place pole on the screen
pole.x = math.sin(pole.angle) * pole.length
pole.y = math.cos(pole.angle) * pole.length
pole.x_scr = screen_x_null + pole.x * screen_scale
pole.y_scr = screen_y_null - pole.y * screen_scale  # y is positive downward on the screen, negative in the model
pole.rect.set_new_position(pole.x_scr, pole.y_scr)

# actuator parameters
motor = Pendulum()
motor.tau = motor_tau  # time constant of the motor
motor.dy = 0  # change of the motor force [N/s]
motor.y = 0   # actual motor force [N]
motor.max_force = motor_max_force  # unit [N]

# pos target
pos_target = Pendulum()
pos_target.position = 0
pos_target.rect = ARectangle(yellow, 5, 5, 40)

force_rect = ARectangle(green, 1, 100, 0)

# create sprite list for moving objects
allSprites.add(pole.rect)
allSprites.add(cart.rect)
allSprites.add(force_rect)
allSprites.add(pos_target.rect)

# create neural network for control
if control_method == 3:
    # inputs:
    #   0: cart position
    #   1: pole angle
    #   2: cart requested position
    #   3: if absolute value of pole angle is smaller than 90°
    # output:
    #   0: requested external force
    if os.path.isfile('control_network.pkl') and 1:
        with open('control_network.pkl', 'rb') as infile:
            control_network = pickle.load(infile)
    else:
        control_network = nn.Network([4, 5, 2, 1], "tanh")
        with open('control_network.pkl', 'wb') as outfile:
            pickle.dump(control_network, outfile, 0)

# load saved trace
if control_method == 4:
    with open('trace.pkl', 'rb') as infile:
        saved_trace = TraceInputOutput()
        saved_trace = pickle.load(infile)

if teach_neural_control == 1 and control_method == 3:
    # inputs:
    #   0: cart position
    #   1: pole angle
    #   2: cart requested position
    #   3: time
    # output:
    #   0: requested external force
    with open('trace.pkl', 'rb') as infile:
        saved_trace = TraceInputOutput()
        saved_trace = pickle.load(infile)
    sample = 0
    for t_sim in saved_trace.input3:
        if abs(saved_trace.input1[sample]) < 45 / 180 * math.pi:
            small_angle = 1
        else:
            small_angle = 0
        input_vector = [np.array([saved_trace.input0[sample], saved_trace.input1[sample],
                                  saved_trace.input2[sample], small_angle])]
        if teach_method_ga:
        #
            teach_nn.teach_network_ga(control_network, input_vector, 0.01, np.array([saved_trace.output0[sample]]))
        else:
            teach_nn.teach_network(control_network, input_vector, 0.0001, 0, np.array([saved_trace.output0[sample]]))
        print('teach_status', '{:3.4f}'.format(t_sim), '/', '{:3.4f}'.format(saved_trace.input3[-1]))
        sample += 1
    with open('control_network.pkl', 'wb') as outfile:
        pickle.dump(control_network, outfile, 0)

# gravitational acceleration constant
g = 9.80665  # m/s^2

# simulation time
t = 0  # unit: s
t_d = 1e-2  # unit: s

# some init values for the first iteration
external_force = 1e-5
requested_position = 0
req_num = 0
trace = TraceInputOutput()
# perform simulation up to the requested time
while t < t_end:
    # calculate requested speed
    requested_speed = (requested_position - pos_target.position) / t_d
    # define the external force to the system on the cart)
    control_output = control_pendulum(pole, cart)
    # change the requested torque based on the motor model time constant
    motor.y = calculate_motor_force(float(control_output))
    external_force = motor.y

    # update system parameters
    # calculate the cart and the pole acceleration values
    accelerations = calculate_accelerations_of_the_system_2(external_force, pole, cart)
    pole = accelerations['pole']
    cart = accelerations['cart']

    # collect trace data to be written to file
    if save_trace_to_file == 1:
        # inputs:
        #   0: cart position
        #   1: pole angle
        #   2: cart requested position
        #   3: time
        # output:
        #   0: requested external force
        trace.input0.append(cart.position)
        trace.input1.append(pole.angle)
        trace.input2.append(requested_position)
        trace.input3.append(t)
        trace.output0.append(control_output)

    draw = 1
    if draw:  # screen update frequency definition
        # remove force from drawing
        allSprites.remove(force_rect)
        # create new sprite for force
        force_scr = abs(int(external_force)) + 1
        force_rect = ARectangle(green, force_scr, 1, 0)
        # add force to the sprites list
        allSprites.add(force_rect)
        # update drawing for the cart
        cart.x_scr = screen_x_null + cart.position * screen_scale
        cart.rect.set_new_position(cart.x_scr, cart.y_scr)

        # update drawing for the pole
        pole.x_scr = screen_x_null + pole.x * screen_scale
        pole.y_scr = screen_y_null - pole.y * screen_scale  # y positive downward on the screen, negative in the model
        pole.rect.set_new_position(pole.x_scr, pole.y_scr)
        pole.rect.rotate_angle(pole.angle)

        # update drawing for the external force signal
        # draw to the left side of the cart if the force is positive
        if external_force > 0:
            force_x = cart.x_scr - cart.length * screen_scale / 2 - force_scr / 2
        # draw to the right side of the cart if the force is negative
        else:
            force_x = cart.x_scr + cart.length * screen_scale / 2 + force_scr / 2
        force_rect.set_new_position(force_x, cart.y_scr)

        # update drawing of the requested position marker
        pos_target.position = requested_position
        pos_target.rect.set_new_position(pos_target.position * screen_scale + screen_x_null,
                                         cart.y_scr - pole.length * screen_scale)

        # clear objects
        # trick: background surface changes are taken over
        # (as the recent changes are exactly where the objects were shown)
        # if the above trick is not good enough than replace the below command (significantly slower)
        allSprites.clear(screen, background)
        # screen.blit(background, (0, 0))

        pygame.event.get()

        # redraw objects
        allSprites.draw(screen)
        # needed to avoid Windows watchdog to react
        pygame.event.get()
        # drawing to screen
        pygame.display.flip()
    # set the fps
    clock.tick(1 / t_d)
    # needed to avoid Windows watchdog to react
    pygame.event.get()
    t = t + t_d
    # if the position is defined as steps, and the next step is now
    if pos_dirac == 1 and t > requested_position_t[req_num]:
        # update the requested position
        requested_position = requested_position_v[req_num]
    # if interpolation is requested between the points
    if pos_dirac == 0:
        # if this is not the first position
        if req_num > 0:
            pos_req_prev = requested_position_v[req_num - 1]
            t_prev = requested_position_t[req_num - 1]
        # if this is the first position
        else:
            pos_req_prev = 0
            t_prev = 0
        # if this is not the last request in the list
        if req_num < len(requested_position_v):
            pos_req_next = requested_position_v[req_num]
            t_next = requested_position_t[req_num]
        # if this is the last request
        else:
            pos_req_next = pos_req_prev
            t_next = t_end
        # if the request is not changed
        if pos_req_next == pos_req_prev:
            requested_position = pos_req_prev
        # if the request is changed, than perform the interpolation
        else:
            requested_position = pos_req_prev + (min(t, t_next) - t_prev) / \
                                                (t_next - t_prev) * (pos_req_next - pos_req_prev)
    # if time has come, jump to the next element in the requested position and time vectors
    if req_num < len(requested_position_v) - 1 and t > requested_position_t[req_num]:
        req_num += 1

if save_trace_to_file == 1:
    with open('trace.pkl', 'wb') as outfile:
        pickle.dump(trace, outfile, 0)
