import pygame

# pygame.init()
pygame.font.init()

class displayHandler:
    
    def __init__(self, width = 800, height = 600):
        
        self.white = (255,255,255)
        self.black = (0,0,0)
        
        self.font = pygame.font.SysFont('arial',12)
        
        self.display = pygame.display.set_mode((width, height))
        self.clock = pygame.time.Clock()
        
        self.left_up_code = None
        self.middle_up_code = None
        self.right_up_code = None
        
        # Arrow keys
        self.A_button_down = None
        self.D_button_down = None
        self.S_button_down = None
        self.W_button_down = None
        self.A_button_up = None
        self.D_button_up = None
        self.S_button_up = None
        self.W_button_up = None
        
        self.key_released =[False]*len(pygame.key.get_pressed())
        self.key_pressed = pygame.key.get_pressed()
        
    def eventHandler(self):
        running = True
        for event in pygame.event.get():
            #The quit event. When the [x] is pressed
            if event.type == pygame.QUIT:
                running = False
            
            #When a key is pressed down
            if event.type == pygame.KEYDOWN:
                if event.key in [pygame.K_ESCAPE, pygame.K_q]:
                    running = False
            
            # Mouse events
            # if event.type == pygame.MOUSEBUTTONUP:
            #     if event.button == 1 and self.left_up_code != None: #left mouse up
            #         self.left_up_code(pygame.mouse.get_pos())
            #     if event.button == 2 and self.middle_up_code != None: #left mouse up
            #         self.middle_up_code(pygame.mouse.get_pos())
            #     if event.button == 3 and self.right_up_code != None: #left mouse up
            #         self.right_up_code(pygame.mouse.get_pos())
            
            # Arrow keys (pressed down)
            # if event.type == pygame.KEYDOWN:
            #     if event.key == pygame.K_a:
            #         self.A_button_down(pygame.mouse.get_pos())
            #     if event.key == pygame.K_d:
            #         self.D_button_down(pygame.mouse.get_pos())
            #     if event.key == pygame.K_w:
            #         self.W_button_down(pygame.mouse.get_pos())
            #     if event.key == pygame.K_s:
            #         self.S_button_down(pygame.mouse.get_pos())

            
            # Arrow keys (released up)
            # if event.type == pygame.KEYUP:
            #     if event.key == pygame.K_a:
            #         self.A_button_up(pygame.mouse.get_pos())
            #     if event.key == pygame.K_d:
            #         self.D_button_up(pygame.mouse.get_pos())
            #     if event.key == pygame.K_w:
            #         self.W_button_up(pygame.mouse.get_pos())
            #     if event.key == pygame.K_s:
            #         self.S_button_up(pygame.mouse.get_pos())          
            
        self.key_released = [i*(True-j) for i,j in zip(self.key_pressed,pygame.key.get_pressed())]
        
        self.key_pressed = pygame.key.get_pressed()
            

        return running

    def drawSquare(self, center, size, color = (0,0,0), width = 0):
        center = (center[0] - size[0]/2,center[1] - size[1]/2)

        pygame.draw.rect(self.display,color, center+ tuple(size), width = width)

    def drawCircle(self, center, radius, color = (0,0,0), width = 0):
        # width: A width of zero makes a solid circle. Else it is non-solid
        pygame.draw.circle(self.display, color,center, radius, width = width)
    
    def drawText(self, message, rect, color = (0,0,0)):

        text = self.font.render(message, True, color)
        self.display.blit(text,rect)
    
    def drawImage(self, path, center, scale = None, angle = None):
        img = pygame.image.load(path)
        if(scale != None):
            img = pygame.transform.scale(img, scale)
        if(angle != None):
            img = pygame.transform.rotate(img, angle)
            
        self.display.blit(img, (center[0] - img.get_width()/2, center[1] - img.get_height()/2))
    
    def close(self):
        pygame.quit()
        
    def update(self, backgroundColor = (255,255,255)):
        pygame.display.flip()
        self.display.fill(backgroundColor)

def drawImage(display, path, center, scale=None, angle=None):
    img = pygame.image.load(path)
    if (scale != None):
        img = pygame.transform.scale(img, scale)
    if (angle != None):
        img = pygame.transform.rotate(img, angle)

    display.blit(img, center)
