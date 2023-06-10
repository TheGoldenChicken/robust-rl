import display
import math
import random

width = 800
height = 600

grid_scale = 100

### Car Constants ###

# Drag coefficient is the amount of air resistance the car has. The drag is calculated by F = K * v^2
K = 0.05

# a: Acceleration is the amount of force that can be applied to the car.
a = 0.8

# m: Mass is the mass of the car
m = 1

# turn_speed: Turn speed is the amount of angle that is added to the car when the A or D key is pressed
turn_speed = 1

### Car Variables ###
# v: Velocity is the current velocity of the car
v = 0

# angle: Angle is the current angle of the car (in degrees)
angle = 0

# pos: Position is the current position of the car
pos = (grid_scale/2,height/2 - grid_scale/2)


render = display.displayHandler(width, height)
running = True

### Car Controls ### 
keys_pressed = [False, False, False, False] # W, A, S, D

### Wind ###

# wind_direction: Wind direction is the direction of the wind (in degrees)
wind_direction = random.randint(0,360)

# wind_force_max: Wind force max is the maximum wind force
wind_force_max = 2.5

# wind_force: Wind force is the force of the wind (in m/s)
wind_force = random.random() * wind_force_max * 2 - wind_force_max

# wind_direction_change: Wind direction change is the amount of degrees that the wind direction changes every frame
wind_direction_change = 5

# wind_force_change: Wind force change is the amount of m/s that the wind force changes every frame
wind_force_change = 0.2




### Key Functions ###

# WASD pressed down
def W_button_down(location):
    keys_pressed[0] = True
    
def A_button_down(location):
    keys_pressed[1] = True

def S_button_down(location):
    keys_pressed[2] = True
    
def D_button_down(location):
    keys_pressed[3] = True

# WASD released up
def W_button_up(location):
    keys_pressed[0] = False
    
def A_button_up(location):
    keys_pressed[1] = False
    
def S_button_up(location):
    keys_pressed[2] = False

def D_button_up(location):
    keys_pressed[3] = False
    
# Set the key events
render.W_button_down = W_button_down
render.A_button_down = A_button_down
render.S_button_down = S_button_down
render.D_button_down = D_button_down
render.W_button_up = W_button_up
render.A_button_up = A_button_up
render.S_button_up = S_button_up
render.D_button_up = D_button_up
    

def game():
    global v, angle, pos, keys_pressed, running, angle
    global turn_speed, a, m, K, grid_scale, width, height
    global render, pos, wind_direction, wind_force, wind_direction_change, wind_force_change
    
    def change_angle():
        global angle, turn_speed, keys_pressed
        if(keys_pressed[1]):
            angle -= turn_speed
        if(keys_pressed[3]):
            angle += turn_speed
    
    while(running):
        ### Physics ###
        # Turn and drive the car if the A or D key is pressed
        if(keys_pressed[0]):
            print("W")
            v += a
            change_angle()
        if(keys_pressed[2]):
            print("S")
            v -= a
            change_angle()
            print(v)
        
        # Calculate the drag force
        F_drag = K * v**2
        
        # Calculate the acceleration
        a_drag = F_drag / m
        
        # Calculate the new velocity
        if(v > 0):
            v -= a_drag
        else:
            v += a_drag
        
        # Calculate the new position
        pos = (pos[0] + v * math.cos(angle*math.pi/180), pos[1] + v * math.sin(angle*math.pi/180))
        
        ### Wind ###
        # Change the wind direction by doing random walks
        wind_direction += wind_direction_change * (2 * (random.random() > 0.5) - 1)
        wind_force += wind_force_change * (2 * (random.random() > 0.5) - 1)
        wind_force = max(0, min(wind_force, wind_force_max))
        
        # Calculate the acceleration
        a_wind = wind_force / m
        
        pos = (pos[0] + a_wind * math.cos(wind_direction*math.pi/180), pos[1] + a_wind * math.sin(wind_direction*math.pi/180))
        
        ### Graphics ###
        # Draw a grid as reference for the board
        for x in range(0,width,grid_scale):
            for y in range(0,height,grid_scale):
                render.drawSquare((x + grid_scale/2,y + grid_scale / 2),
                                  (grid_scale,grid_scale), (50,50,50), width = 5)
        
        for x in range(0,int(width/grid_scale)):
            for y in [int(height/grid_scale)-2,int(height/grid_scale)-1]:
                render.drawImage("WaterTile.png", (x*grid_scale,y*grid_scale), (grid_scale,grid_scale))
        
        # Draw the car
        render.drawImage("Car.png", (pos[0] - grid_scale/4, pos[1] - grid_scale/4), (grid_scale/2,grid_scale/2), -angle)
        
        # Draw the wind direction
        render.drawImage("WindArrow.png", (width - 3*grid_scale/4, grid_scale/4), (grid_scale/2,grid_scale/2), -wind_direction+45)
        render.drawText("Wind Force: " + str(round(wind_force,2)), (width - 5*grid_scale/6, grid_scale), (255,255,255))
        
        render.update((0,0,0))
        
        running = render.eventHandler()
        


if __name__ == "__main__":

    game()
    
    