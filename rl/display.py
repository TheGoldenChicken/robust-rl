import pygame

# pygame.init()
pygame.font.init()

class displayHandler:
    
    def __init__(self, width = 800, height = 600):
        
        self.white = (255,255,255)
        self.black = (0,0,0)
        
        self.font_size = 12
        self.font = pygame.font.SysFont('arial',self.font_size)
        
        
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
                self.close()
                running = False        
            
        # Detect key releases
        self.key_released = [i*(True-j) for i,j in zip(self.key_pressed,pygame.key.get_pressed())]
        
        # Detect key presses
        self.key_pressed = pygame.key.get_pressed()
        
        if(self.key_released[pygame.K_ESCAPE] or self.key_released[pygame.K_q]):
            self.close()
            running = False

        return running

    def draw_square(self, center, size, color = (0,0,0), width = 0, edge_color = (100,100,100)):
        top_left = (center[0] - size[0]/2,center[1] - size[1]/2)

        pygame.draw.rect(self.display, color, top_left+ tuple(size))
        
        if(width > 0): pygame.draw.rect(self.display,edge_color, top_left+ tuple(size), width = width)
            
    def draw_sphere(self, center, radius, color = (0,0,0), width = 0, edge_color = (100,100,100)):
        # width: A width of zero makes a solid circle. Else it is non-solid
        pygame.draw.circle(self.display, color, center, radius)
        
        if(width > 0): pygame.draw.circle(self.display, edge_color, center, radius, width = width)
        
        
    
    def draw_text(self, message, rect, color = (0,0,0), align="top-left", angle = None, font = 'arial', font_size = None):
        
        self.set_font(font, font_size)
        text = self.font.render(message, True, color)
        
        if(angle != None):
            text = pygame.transform.rotate(text, angle)
        
        # Match case on align with the following options
        match (align):
            case "top-left":
                rect = rect
            case "top-right":
                rect = (rect[0] - text.get_width(), rect[1])
            case "bottom-left":
                rect = (rect[0], rect[1] - text.get_height())
            case "bottom-right":
                rect = (rect[0] - text.get_width(), rect[1] - text.get_height())
            case "center":
                rect = (rect[0] - text.get_width()/2, rect[1] - text.get_height()/2)
            case "center-left":
                rect = (rect[0], rect[1] - text.get_height()/2)
            case "center-right":
                rect = (rect[0] - text.get_width(), rect[1] - text.get_height()/2)
            case "center-top":
                rect = (rect[0] - text.get_width()/2, rect[1])
            case "center-bottom":
                rect = (rect[0] - text.get_width()/2, rect[1] - text.get_height())
        
        
        self.display.blit(text,rect)
    
    def draw_image(self, path, center, scale = None, angle = None):
        img = pygame.image.load(path)
        if(scale != None):
            img = pygame.transform.scale(img, scale)
        if(angle != None):
            img = pygame.transform.rotate(img, angle)
            
        self.display.blit(img, (center[0] - img.get_width()/2, center[1] - img.get_height()/2))
    
    def draw_polygon(self, points, color = (0,0,0), width = 0, edge_color = (100,100,100)):
        pygame.draw.polygon(self.display, color, points)
        
        if(width > 0): pygame.draw.polygon(self.display, edge_color, points, width = width)
    
    def set_font(self, font_name = "arial", font_size = None):
        if font_size == None: size = self.font_size
        else: self.font_size = font_size
        self.font = pygame.font.SysFont(font_name, self.font_size)
    
    def close(self):
        pygame.quit()
        
    def update(self, backgroundColor = (255,255,255)):
        pygame.display.flip()
        self.display.fill(backgroundColor)