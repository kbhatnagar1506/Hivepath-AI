
const express = require('express');
const { createProxyMiddleware } = require('http-proxy-middleware');
const cors = require('cors');

const app = express();
const PORT = process.env.PORT || 8080;

// Enable CORS
app.use(cors());

// Proxy API requests to backend
app.use('/api', createProxyMiddleware({
    target: 'http://localhost:8000',
    changeOrigin: true,
    pathRewrite: {
        '^/api': '/api/v1'
    }
}));

// Serve frontend (if local build available)
app.use('/', express.static('dist'));

// Fallback to external frontend
app.get('*', (req, res) => {
    res.redirect('https://fleet-flow-7189cccb.base44.app' + req.path);
});

app.listen(PORT, () => {
    console.log(`🚀 Integrated server running on port ${PORT}`);
    console.log(`📡 API proxy: http://localhost:${PORT}/api`);
    console.log(`🌐 Frontend: https://fleet-flow-7189cccb.base44.app`);
});
