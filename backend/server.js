import express from 'express'
import cors from 'cors'
import multer from 'multer'
import dotenv from 'dotenv'
import path from 'path'
import fs from 'fs'
import { fileURLToPath } from 'url'
import { spawn } from 'child_process'
import { tryOnnxInference } from './inference/onnx_inference.js'
// Environment variables used:
//  - PORT: server port
//  - CORS_ORIGINS: comma separated list of allowed origins
//  - PIPELINE_PATH: optional path to the serialized Python pipeline
//  - PYTHON_BIN: Python executable used for inference
//  - INFERENCE_RUNTIME: onnx (default, low-memory) or python

// ES module compatibility
const __filename = fileURLToPath(import.meta.url)
const __dirname = path.dirname(__filename)

dotenv.config()

const app = express()
const PORT = process.env.PORT || 5000

// Middleware
app.use(cors({
  origin: process.env.CORS_ORIGINS?.split(',') || ['http://localhost:5173'],
  credentials: true
}))

app.use(express.json())
app.use(express.urlencoded({ extended: true }))

// File-based user tracking routes
app.post('/api/users/track', (req, res) => {
  try {
    const {
      sessionId,
      userAgent,
      language,
      platform,
      cookieEnabled,
      doNotTrack,
      screenWidth,
      screenHeight,
      screenColorDepth,
      windowWidth,
      windowHeight,
      timezone,
      canvasFingerprint,
      sessionStart,
      webGL,
      touchSupport,
      localStorage,
      sessionStorage,
      connectionType
    } = req.body;

    // Get IP address from request
    const ipAddress = req.ip || req.connection.remoteAddress || 
                     req.socket.remoteAddress || 
                     (req.connection.socket ? req.connection.socket.remoteAddress : null);

    const trackingFilePath = path.join(__dirname, 'data/userTracking.json');
    const countFilePath = path.join(__dirname, 'data/userCount.json');
    
    // Load existing tracking data
    let trackingData = [];
    if (fs.existsSync(trackingFilePath)) {
      trackingData = JSON.parse(fs.readFileSync(trackingFilePath, 'utf8'));
    }

    // Check if user already exists
    const existingUserIndex = trackingData.findIndex(user => user.sessionId === sessionId);

    if (existingUserIndex !== -1) {
      // Update existing user
      trackingData[existingUserIndex].visitCount += 1;
      trackingData[existingUserIndex].lastVisit = new Date().toISOString();
      
      fs.writeFileSync(trackingFilePath, JSON.stringify(trackingData, null, 2));
      
      res.json({ 
        success: true, 
        message: "User tracking updated",
        visitCount: trackingData[existingUserIndex].visitCount,
        isReturning: true
      });
    } else {
      // Create new user tracking record
      const newTracking = {
        sessionId,
        userAgent,
        language,
        platform,
        cookieEnabled,
        doNotTrack,
        screenWidth,
        screenHeight,
        screenColorDepth,
        windowWidth,
        windowHeight,
        timezone,
        canvasFingerprint,
        sessionStart,
        webGL,
        touchSupport,
        localStorage,
        sessionStorage,
        connectionType,
        ipAddress,
        visitCount: 1,
        lastVisit: new Date().toISOString(),
        createdAt: new Date().toISOString()
      };

      trackingData.push(newTracking);
      fs.writeFileSync(trackingFilePath, JSON.stringify(trackingData, null, 2));

      // Also increment the general user count
      let countData = { count: 0 };
      if (fs.existsSync(countFilePath)) {
        countData = JSON.parse(fs.readFileSync(countFilePath, 'utf8'));
      }
      countData.count += 1;
      fs.writeFileSync(countFilePath, JSON.stringify(countData, null, 2));

      res.json({ 
        success: true, 
        message: "User tracked successfully",
        visitCount: 1,
        isReturning: false,
        totalUsers: countData.count
      });
    }

  } catch (error) {
    console.error("Error tracking user:", error);
    res.status(500).json({ success: false, message: "Tracking error" });
  }
});

// Get user analytics
app.get('/api/users/analytics', (req, res) => {
  try {
    const trackingFilePath = path.join(__dirname, 'data/userTracking.json');
    
    if (!fs.existsSync(trackingFilePath)) {
      return res.json({
        totalUsers: 0,
        todayUsers: 0,
        browserStats: [],
        timezoneStats: []
      });
    }

    const trackingData = JSON.parse(fs.readFileSync(trackingFilePath, 'utf8'));
    const today = new Date();
    today.setHours(0, 0, 0, 0);

    const totalUsers = trackingData.length;
    const todayUsers = trackingData.filter(user => 
      new Date(user.createdAt) >= today
    ).length;

    // Browser/Platform stats
    const platformCounts = {};
    trackingData.forEach(user => {
      platformCounts[user.platform] = (platformCounts[user.platform] || 0) + 1;
    });
    const browserStats = Object.entries(platformCounts).map(([platform, count]) => ({
      _id: platform,
      count
    })).sort((a, b) => b.count - a.count);

    // Timezone stats
    const timezoneCounts = {};
    trackingData.forEach(user => {
      timezoneCounts[user.timezone] = (timezoneCounts[user.timezone] || 0) + 1;
    });
    const timezoneStats = Object.entries(timezoneCounts).map(([timezone, count]) => ({
      _id: timezone,
      count
    })).sort((a, b) => b.count - a.count);

    res.json({
      totalUsers,
      todayUsers,
      browserStats,
      timezoneStats
    });
  } catch (error) {
    console.error("Error fetching analytics:", error);
    res.status(500).json({ success: false, message: "Analytics error" });
  }
});

// Get current user count
app.get('/api/users/count', (req, res) => {
  try {
    const filePath = path.join(__dirname, 'data/userCount.json');
    if (fs.existsSync(filePath)) {
      const data = JSON.parse(fs.readFileSync(filePath, 'utf8'));
      res.json({ count: data.count });
    } else {
      res.json({ count: 0 });
    }
  } catch (error) {
    console.error('Error reading user count:', error);
    res.status(500).json({ success: false, message: 'Server error' });
  }
});

// Static file serving for downloads
app.use('/downloads', express.static(path.join(__dirname, 'storage/downloads')))

// Multer configuration for file uploads
const storage = multer.diskStorage({
  destination: (req, file, cb) => {
    const uploadDir = path.join(__dirname, 'storage/uploads')
    if (!fs.existsSync(uploadDir)) {
      fs.mkdirSync(uploadDir, { recursive: true })
    }
    cb(null, uploadDir)
  },
  filename: (req, file, cb) => {
    const uniqueSuffix = Date.now() + '-' + Math.round(Math.random() * 1E9)
    cb(null, file.fieldname + '-' + uniqueSuffix + path.extname(file.originalname))
  }
})

const upload = multer({ 
  storage: storage,
  limits: {
    fileSize: 10 * 1024 * 1024 // 10MB limit
  },
  fileFilter: (req, file, cb) => {
    if (file.mimetype.startsWith('image/')) {
      cb(null, true)
    } else {
      cb(new Error('Only image files are allowed!'), false)
    }
  }
})

// Routes

// Get available models
app.get('/api/models', (req, res) => {
  try {
    const modelsPath = path.join(__dirname, 'data/models.json')
    const models = JSON.parse(fs.readFileSync(modelsPath, 'utf8'))
    res.json(models)
  } catch (error) {
    console.error('Error reading models:', error)
    res.status(500).json({ error: 'Failed to load models' })
  }
})

// Get available downloads
app.get('/api/downloads', (req, res) => {
  try {
    const downloadsPath = path.join(__dirname, 'data/downloads.json')
    const downloads = JSON.parse(fs.readFileSync(downloadsPath, 'utf8'))
    res.json(downloads)
  } catch (error) {
    console.error('Error reading downloads:', error)
    res.status(500).json({ error: 'Failed to load downloads' })
  }
})

// Download specific file
app.get('/api/downloads/:filename', (req, res) => {
  const filename = req.params.filename
  const filePath = path.join(__dirname, 'storage/downloads', filename)
  
  if (fs.existsSync(filePath)) {
    res.download(filePath)
  } else {
    res.status(404).json({ error: 'File not found' })
  }
})

// Predict endpoint - handles image analysis
app.post('/api/predict', (req, res) => {
  res.status(410).json({ error: 'This endpoint was retired. Use POST /api/analyze with one image.' })
})

// New single-image analysis using ONNX runtime (Node-only)
app.post('/api/analyze', upload.single('image'), async (req, res) => {
  try {
    if (!req.file) {
      return res.status(400).json({ error: 'Image file is required (field name: image)' })
    }
    const usePython = process.env.INFERENCE_RUNTIME === 'python'
    const result = usePython
      ? await runPythonPipeline(req.file.path)
      : await tryOnnxInference(req.file.path)
    if (!result.success) {
      return res.status(500).json({ error: 'Model inference failed', details: result.error })
    }
    return res.json({ ...result, runtime: usePython ? 'python-pipeline' : 'onnx-node' })
  } catch (err) {
    console.error('Analyze route error:', err)
    res.status(500).json({ error: 'Server error during analysis' })
  }
})

function runPythonPipeline(imagePath) {
  const pipelinePath = process.env.PIPELINE_PATH || path.join(__dirname, 'models', 'phytoplankton_full_pipeline.pkl')
  const pythonBin = process.env.PYTHON_BIN || 'python'
  const scriptPath = path.join(__dirname, 'inference', 'run_pipeline.py')

  return new Promise((resolve, reject) => {
    const child = spawn(pythonBin, [scriptPath, '--pipeline', pipelinePath, '--image', imagePath, '--topk', '10'], {
      cwd: __dirname,
      env: process.env
    })
    let stdout = ''
    let stderr = ''
    child.stdout.on('data', chunk => { stdout += chunk.toString() })
    child.stderr.on('data', chunk => { stderr += chunk.toString() })
    child.on('error', reject)
    child.on('close', code => {
      let result
      try { result = JSON.parse(stdout.trim()) } catch { return reject(new Error(stderr || 'Invalid inference response')) }
      if (code !== 0 || !result.success) return reject(new Error(result.error || stderr || 'Pipeline inference failed'))
      resolve(result)
    })
  })
}

// Health check endpoint
app.get('/api/health', (req, res) => {
  res.json({ 
    status: 'healthy', 
    timestamp: new Date().toISOString(),
    demoMode: false,
    pipelineConfigured: fs.existsSync(path.join(__dirname, 'models', 'phytoplankton_full_pipeline.onnx')),
    runtime: process.env.INFERENCE_RUNTIME === 'python' ? 'python-pipeline' : 'onnx-node'
  })
})

// Error handling middleware
app.use((error, req, res, next) => {
  if (error instanceof multer.MulterError) {
    if (error.code === 'LIMIT_FILE_SIZE') {
      return res.status(400).json({ error: 'File too large. Maximum size is 10MB.' })
    }
  }
  
  console.error('Server error:', error)
  res.status(500).json({ error: 'Internal server error' })
})

// 404 handler
app.use('*', (req, res) => {
  res.status(404).json({ error: 'Route not found' })
})

// Start server
app.listen(PORT, () => {
  console.log(`🚀 Marine Microscopy API Server running on port ${PORT}`)
  console.log(`📊 Demo mode: ${process.env.ENABLE_DEMO_MODE === 'true' ? 'ENABLED' : 'DISABLED'}`)
  console.log(`🌐 CORS origins: ${process.env.CORS_ORIGINS || 'http://localhost:5173'}`)
  
  // Create required directories
  const dirs = ['storage/uploads', 'storage/downloads', 'data']
  dirs.forEach(dir => {
    const fullPath = path.join(__dirname, dir)
    if (!fs.existsSync(fullPath)) {
      fs.mkdirSync(fullPath, { recursive: true })
      console.log(`📁 Created directory: ${dir}`)
    }
  })
})
