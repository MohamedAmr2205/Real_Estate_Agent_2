CREATE TABLE Agent (
    agent_id INTEGER PRIMARY KEY,
    full_name VARCHAR(100) NOT NULL,
    email VARCHAR(100) UNIQUE,
    phone VARCHAR(20),
    license_number VARCHAR(50) UNIQUE,
    hire_date DATE,
    role VARCHAR(50)
);



CREATE TABLE Customer (
    customer_id INTEGER PRIMARY KEY,
    full_name VARCHAR(100) NOT NULL,
    email VARCHAR(100) UNIQUE,
    phone VARCHAR(20),
    customer_type VARCHAR(30),
    registration_date DATE
);



CREATE TABLE Property (
    property_id INTEGER PRIMARY KEY,
    title VARCHAR(150),
    address VARCHAR(200),
    city VARCHAR(50),
    property_type VARCHAR(50),
    bedrooms INTEGER,
    bathrooms INTEGER,
    area_sqft DECIMAL(10,2),
    price DECIMAL(12,2),
    status VARCHAR(30),

    owner_id INTEGER,
    agent_id INTEGER,

    FOREIGN KEY(owner_id) REFERENCES Customer(customer_id),
    FOREIGN KEY(agent_id) REFERENCES Agent(agent_id)
);



CREATE TABLE Appointment (
    appointment_id INTEGER PRIMARY KEY,

    property_id INTEGER,
    customer_id INTEGER,
    agent_id INTEGER,

    appointment_date DATETIME,
    status VARCHAR(30),

    FOREIGN KEY(property_id) REFERENCES Property(property_id),
    FOREIGN KEY(customer_id) REFERENCES Customer(customer_id),
    FOREIGN KEY(agent_id) REFERENCES Agent(agent_id)
);


CREATE TABLE Offer (
    offer_id INTEGER PRIMARY KEY,

    property_id INTEGER,
    customer_id INTEGER,

    offer_amount DECIMAL(12,2),
    offer_date DATE,
    status VARCHAR(30),

    FOREIGN KEY(property_id) REFERENCES Property(property_id),
    FOREIGN KEY(customer_id) REFERENCES Customer(customer_id)
);


--
CREATE TABLE Contract (
    contract_id INTEGER PRIMARY KEY,

    property_id INTEGER,
    customer_id INTEGER,
    agent_id INTEGER,

    contract_date DATE,
    contract_type VARCHAR(20),
    total_amount DECIMAL(12,2),

    FOREIGN KEY(property_id) REFERENCES Property(property_id),
    FOREIGN KEY(customer_id) REFERENCES Customer(customer_id),
    FOREIGN KEY(agent_id) REFERENCES Agent(agent_id)
);



CREATE TABLE Maintenance_Request (
    request_id INTEGER PRIMARY KEY,

    property_id INTEGER,
    customer_id INTEGER,

    description TEXT,
    request_date DATE,
    status VARCHAR(30),

    FOREIGN KEY(property_id) REFERENCES Property(property_id),
    FOREIGN KEY(customer_id) REFERENCES Customer(customer_id)
);



CREATE TABLE Property_Document (
    document_id INTEGER PRIMARY KEY,

    property_id INTEGER,

    document_type VARCHAR(50),
    document_path VARCHAR(255),
    upload_date DATE,

    FOREIGN KEY(property_id) REFERENCES Property(property_id)
);