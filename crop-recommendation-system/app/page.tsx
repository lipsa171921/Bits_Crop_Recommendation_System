export default function HomePage() {
  return (
    <div className="min-h-screen bg-gradient-to-br from-green-50 to-green-100 flex items-center justify-center p-4">
      <div className="max-w-4xl mx-auto text-center space-y-8">
        <div className="space-y-4">
          <h1 className="text-5xl font-bold text-green-800 mb-4">🌾 Crop Recommendation System</h1>
          <p className="text-xl text-green-700 mb-8">AI-Powered Agricultural Decision Support System</p>
        </div>

        <div className="bg-white rounded-2xl shadow-xl p-8 border border-green-200">
          <div className="space-y-6">
            <div className="text-center">
              <div className="text-6xl mb-4">🚀</div>
              <h2 className="text-3xl font-bold text-gray-800 mb-4">Launch Streamlit Application</h2>
              <p className="text-gray-600 mb-6">
                This crop recommendation system runs on Streamlit with advanced ML models and interactive
                visualizations.
              </p>
            </div>

            <div className="bg-gray-50 rounded-lg p-6 text-left">
              <h3 className="text-lg font-semibold text-gray-800 mb-4">📋 How to Run:</h3>
              <div className="space-y-3">
                <div className="flex items-start space-x-3">
                  <span className="bg-green-500 text-white rounded-full w-6 h-6 flex items-center justify-center text-sm font-bold">
                    1
                  </span>
                  <div>
                    <p className="font-medium text-gray-800">Install Dependencies</p>
                    <code className="bg-gray-200 px-2 py-1 rounded text-sm">
                      cd scripts && pip install -r requirements.txt
                    </code>
                  </div>
                </div>
                <div className="flex items-start space-x-3">
                  <span className="bg-green-500 text-white rounded-full w-6 h-6 flex items-center justify-center text-sm font-bold">
                    2
                  </span>
                  <div>
                    <p className="font-medium text-gray-800">Run Streamlit App</p>
                    <code className="bg-gray-200 px-2 py-1 rounded text-sm">streamlit run streamlit_app.py</code>
                  </div>
                </div>
                <div className="flex items-start space-x-3">
                  <span className="bg-green-500 text-white rounded-full w-6 h-6 flex items-center justify-center text-sm font-bold">
                    3
                  </span>
                  <div>
                    <p className="font-medium text-gray-800">Open Browser</p>
                    <p className="text-gray-600">
                      Navigate to <code className="bg-gray-200 px-2 py-1 rounded text-sm">http://localhost:8501</code>
                    </p>
                  </div>
                </div>
              </div>
            </div>

            <div className="grid grid-cols-1 md:grid-cols-3 gap-4 mt-8">
              <div className="bg-green-50 p-4 rounded-lg border border-green-200">
                <div className="text-2xl mb-2">🤖</div>
                <h4 className="font-semibold text-green-800">AI Models</h4>
                <p className="text-sm text-green-600">6 ML algorithms with 95%+ accuracy</p>
              </div>
              <div className="bg-green-50 p-4 rounded-lg border border-green-200">
                <div className="text-2xl mb-2">📊</div>
                <h4 className="font-semibold text-green-800">Interactive Charts</h4>
                <p className="text-sm text-green-600">Plotly visualizations and data analysis</p>
              </div>
              <div className="bg-green-50 p-4 rounded-lg border border-green-200">
                <div className="text-2xl mb-2">🌾</div>
                <h4 className="font-semibold text-green-800">22 Crop Types</h4>
                <p className="text-sm text-green-600">Comprehensive crop recommendations</p>
              </div>
            </div>

            <div className="bg-blue-50 border border-blue-200 rounded-lg p-4 mt-6">
              <div className="flex items-center space-x-2">
                <div className="text-blue-500">ℹ️</div>
                <p className="text-blue-800 font-medium">Note:</p>
              </div>
              <p className="text-blue-700 mt-2">
                The ML models will be trained automatically when you first run the Streamlit app. This may take a few
                minutes on the initial launch.
              </p>
            </div>
          </div>
        </div>

        <div className="text-center">
          <p className="text-gray-500">Built with ❤️ for sustainable agriculture</p>
        </div>
      </div>
    </div>
  )
}
