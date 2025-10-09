-- define logic of db and what we have to save 
CREATE TABLE IF NOT EXISTS users (
    id SERIAL PRIMARY KEY,
    username VARCHAR(50) UNIQUE NOT NULL,
    email VARCHAR(100) UNIQUE NOT NULL,
    password_hash VARCHAR(255) NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS images (
    id SERIAL PRIMARY KEY,
    user_id INT REFERENCES users(id) ON DELETE CASCADE,
    image_path VARCHAR(255) NOT NULL,
    uploaded_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS classifications (
    id SERIAL PRIMARY KEY,
    image_id INT REFERENCES images(id) ON DELETE CASCADE,
    classification VARCHAR(50) NOT NULL,
    confidence FLOAT NOT NULL,
    classified_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS facesCelebrities (
    id SERIAL PRIMARY KEY,
    image_id INT REFERENCES images(id) ON DELETE CASCADE,
    celebrity_name VARCHAR(100) NOT NULL,
    detected_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);