# Critical Patient Management System (CPMS)

A comprehensive healthcare monitoring and analysis platform that leverages machine learning for patient condition assessment and prediction.

## Features

- **AI-Powered Analysis**: Multiple machine learning algorithms for accurate patient condition assessment
- **Real-time Monitoring**: Continuous tracking of vital signs with immediate alerts
- **Secure Data Management**: HIPAA-compliant data storage and transmission
- **Multi-algorithm Support**: Various ML algorithms for robust predictions
- **Interactive Visualization**: Real-time charts and graphs for data analysis
- **Batch Processing**: Efficient handling of multiple patient records
- **User Authentication**: Secure login system with role-based access

## Technical Stack

- **Backend**: Python Flask
- **Frontend**: HTML5, CSS3, JavaScript
- **Machine Learning**: Scikit-learn
- **Data Visualization**: Chart.js
- **UI Framework**: Bootstrap 5
- **Icons**: Font Awesome

## Installation

1. Clone the repository:
```bash
git clone [repository-url]
cd critical-patient-management-system
```

2. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Run the application:
```bash
python app.py
```

The application will be available at `http://127.0.0.1:5000`

## Usage

### Authentication
- Default credentials:
  - Username: `admin`
  - Password: `admin123`

### Patient Data Analysis
1. Navigate to the System page
2. Upload patient data in CSV format
3. Select analysis algorithm:
   - Naive Bayes
   - Logistic Regression
   - KNN
   - Decision Tree
   - Random Forest
   - Gradient Boosting
   - Neural Network
   - Various Bagging algorithms

### Patient Vitals Prediction
1. Upload patient vitals file (CSV or TXT)
2. System will analyze and predict patient status
3. View results in interactive table format
4. Get final status based on multiple algorithm predictions

## Features in Detail

### Data Analysis
- Multiple ML algorithms for comprehensive analysis
- Real-time performance metrics
- Interactive visualization of results
- Comparative analysis across algorithms

### Patient Monitoring
- Batch processing of patient records
- Real-time status updates
- Color-coded status indicators
- Comprehensive vital sign analysis

### Security
- Secure user authentication
- Role-based access control
- Encrypted data transmission
- HIPAA-compliant data handling

## System Requirements

- Python 3.8 or higher
- Modern web browser (Chrome, Firefox, Safari, Edge)
- 4GB RAM minimum
- 500MB disk space

## Contributing

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Support

For support, please contact:
- Email: support@cpms.com
- Phone: +1 (555) 123-4567

## Acknowledgments

- Scikit-learn team for ML algorithms
- Bootstrap team for UI framework
- Chart.js team for visualization tools
- Font Awesome for icons

## Deployment Instructions

1. Fork this repository to your GitHub account
2. Go to [Netlify](https://www.netlify.com/) and sign up/login
3. Click "New site from Git"
4. Choose GitHub and select your forked repository
5. Configure the build settings:
   - Build command: `pip install -r requirements.txt`
   - Publish directory: `templates`
6. Click "Deploy site"

## Local Development

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Run the application:
```bash
python app.py
```

3. Access the application at `http://localhost:5000`

## Features

- Upload and analyze patient data
- Multiple machine learning algorithms
- Real-time patient vital monitoring
- Performance metrics visualization
- Batch prediction capabilities 