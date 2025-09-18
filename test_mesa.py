import os
os.environ['PYOPENGL_PLATFORM'] = 'osmesa'

try:
    from OpenGL import GL
    print("PyOpenGL with OSMesa: SUCCESS")
except Exception as e:
    print(f"PyOpenGL failed: {e}")
    
try:
    import OpenGL
    print(f"OpenGL version: {OpenGL.__version__}")
    print(f"Platform: {OpenGL.platform.PLATFORM}")
except Exception as e:
    print(f"OpenGL info failed: {e}")