INSERT INTO Agent 
(agent_id, full_name, email, phone, license_number, hire_date, role)
VALUES
(1, 'Ahmed Hassan', 'ahmed.hassan@cornerstone.com', '01015376543', 'LIC1001', '2025-01-15', 'Senior Agent'),
(2, 'Sara Mohamed', 'sara.mohamed@cornerstone.com', '01023765555', 'LIC1002', '2025-03-10', 'Real Estate Agent'),
(3, 'Omar Ali', 'omar.ali@cornerstone.com', '01024334355', 'LIC1003', '2025-06-20', 'Property Manager'),
(4, 'Khaled Fathy', 'khaled.fathy@cornerstone.com', '01098765432', 'LIC1004', '2025-02-01', 'Broker');


INSERT INTO Customer
(customer_id, full_name, email, phone, customer_type, registration_date)
VALUES
(1, 'Mohamed Adel', 'mohamed.adel@gmail.com', '01112332112', 'Buyer', '2026-01-10'),
(2, 'Mona Khaled', 'mona.khaled@gmail.com', '01123232456', 'Seller', '2024-07-05'),
(3, 'Youssef Samir', 'youssef.samir@gmail.com', '01111133243', 'Seller', '2025-09-25'),
(4, 'Nour Ahmed', 'nour.ahmed@gmail.com', '01165678909', 'Seller', '2025-10-12'),
(5, 'Ali Hassan', 'ali.hassan@gmail.com', '01234567890', 'Buyer', '2026-05-01');


INSERT INTO Property
(property_id, title, address, city, property_type, bedrooms, bathrooms, area_sqft, price, status, owner_id, agent_id)
VALUES

(1, 'Luxury Villa in Smouha', 'Street 10, Smouha', 'Alexandria', 'Villa', 5, 4, 3000, 5000000, 'Available', 2, 1),
(2, 'Modern Apartment', 'Mostafa Kamel Street', 'Alexandria', 'Apartment', 3, 2, 1500, 1800000, 'Pending', 4, 2),
(3, 'Office Building', 'Corniche Road', 'Alexandria', 'Office', 0, 2, 2000, 3500000, 'Sold', 3, 3),
(4, 'Small Apartment', 'Loran', 'Alexandria', 'Apartment', 2, 1, 900, 900000, 'Available', 2, 2),
(5, 'Beach House', 'Miami Street', 'Alexandria', 'Villa', 4, 3, 2500, 4200000, 'Withdrawn', 2, 1);



INSERT INTO Appointment
(appointment_id, property_id, customer_id, agent_id, appointment_date, status)
VALUES
(1, 1, 1, 1, '2026-07-29 10:00:00', 'Scheduled'),
(2, 2, 1, 2, '2026-07-20 14:30:00', 'Completed');

INSERT INTO Offer
(offer_id, property_id, customer_id, offer_amount, offer_date, status)
VALUES

(1, 1, 1, 4800000, '2026-07-20', 'Pending'),
(2, 2, 1, 1750000, '2026-07-21', 'Rejected'),
(3, 1, 5, 3000000, '2026-07-25', 'Pending');

INSERT INTO Contract
(contract_id, property_id, customer_id, agent_id, contract_date, contract_type, total_amount)
VALUES

(1, 3, 1, 3, '2026-07-15', 'Sale', 3500000);



INSERT INTO Maintenance_Request
(request_id, property_id, customer_id, description, request_date, status)
VALUES
(1, 1, 2, 'Air conditioning maintenance required', '2026-06-25', 'Closed'),
(2, 2, 4, 'Water leakage issue', '2026-04-23', 'Open');

INSERT INTO Property_Document
(document_id, property_id, document_type, document_path, upload_date)
VALUES
(1, 1, 'Ownership Document', '/documents/villa_owner.pdf', '2024-05-15'),
(2, 2, 'Ownership Document', '/documents/apartment_owner.pdf', '2025-02-10'),
(3, 3, 'Ownership Document', '/documents/office_owner.pdf', '2025-09-20'),
(4, 4, 'Ownership Document', '/documents/small_apartment_owner.pdf', '2026-05-01'),
(5, 5, 'Ownership Document', '/documents/beach_house_owner.pdf', '2024-01-01');