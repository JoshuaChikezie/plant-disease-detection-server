#!/usr/bin/env python3
"""
Run Plant Disease Detection Server with TensorFlow
This script uses the working TensorFlow environment from C:\tfproject
"""

import sys
import os

# Add the working TensorFlow environment to the path
tf_venv_path = r"C:\tfproject\venv\Lib\site-packages"
if tf_venv_path not in sys.path:
    sys.path.insert(0, tf_venv_path)

# Add the current directory to the path for local imports
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)

# Also add the Scripts directory for executables
tf_scripts_path = r"C:\tfproject\venv\Scripts"
if tf_scripts_path not in os.environ['PATH']:
    os.environ['PATH'] = tf_scripts_path + os.pathsep + os.environ['PATH']

try:
    # Test TensorFlow import
    import tensorflow as tf
    print(f"✅ TensorFlow {tf.__version__} imported successfully!")
    
    # Test other essential packages
    import numpy as np
    import cv2
    import sklearn
    print(f"✅ NumPy {np.__version__} imported successfully!")
    print(f"✅ OpenCV {cv2.__version__} imported successfully!")
    print(f"✅ Scikit-learn {sklearn.__version__} imported successfully!")
    
    # Now import and run the main server
    print("🚀 Starting Plant Disease Detection Server with TensorFlow...")
    
    # Import the main server components with absolute imports
    try:
        from api.main import app
        print("✅ Main API imported successfully!")
    except ImportError as e:
        print(f"❌ API import error: {e}")
        print("Trying alternative import method...")
        
        # Try to import the app directly
        import importlib.util
        spec = importlib.util.spec_from_file_location("main", "api/main.py")
        main_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(main_module)
        app = main_module.app
        print("✅ API imported using alternative method!")
    
    # Start the server
    import uvicorn
    print("🌐 Starting server on http://localhost:8000")
    print("📚 API documentation will be available at http://localhost:8000/docs")
    
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )
    
except ImportError as e:
    print(f"❌ Import error: {e}")
    print("Please ensure TensorFlow is installed in C:\\tfproject\\venv")
    sys.exit(1)
except Exception as e:
    print(f"❌ Error: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)
