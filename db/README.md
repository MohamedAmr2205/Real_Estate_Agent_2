# Database Documentation

## Database Engine

SQLite

## Why SQLite?

We selected SQLite as the database engine because it is simple, lightweight, and easy to set up.
It does not require a separate database server, which makes it suitable for local development and testing.

SQLite also provides the required SQL features for creating tables, defining relationships, and executing queries.
It can be easily connected with the MCP Server to perform database operations.


## Database Entities

The database contains the following entities:

- Agent:
Stores information about real estate agents, including their names, contact information, licenses, and roles.
- Customer:
Stores customer information such as buyers and sellers.
- Property:
Stores details about real estate properties including location, type, price, status, ownership information, and assigned agents.
- Appointment:
Stores property viewing appointments between customers, agents, and properties.
- Offer:
Stores purchase offers submitted by customers for available properties.
- Contract:
Stores sale and rental contract information between customers, properties, and agents.
- Maintenance_Request:
Stores maintenance requests related to properties.
- Property_Document:
Stores documents related to properties such as ownership documents.


## Database Relationships

The main relationships between entities are:

- One Agent can manage many Properties.
- One Customer can own many Properties as a seller.
- One Property can have many Appointments.
- One Customer can create many Appointments.
- One Property can receive many Offers.
- One Customer can submit many Offers.
- One Property can have many Contracts.
- One Agent can manage many Contracts.
- One Property can have many Maintenance Requests.
- One Property can have many Documents.


## Test Cases

The database was tested using different scenarios to make sure that the schema, relationships, and business rules work correctly.


## Normal Cases

### 1. Agent and Property Relationship

A real estate agent can manage multiple properties.
Test:
Retrieve all properties assigned to each agent and verify that the relationship works correctly.


### 2. Customer and Appointment Relationship

Customers can schedule appointments to view properties.
Test:
Retrieve appointment details with the related customer, property, and agent information.


### 3. Property and Offer Relationship

Customers can submit offers for available properties.
Test:
Retrieve all offers associated with each property and check their status.


### 4. Contract Creation

Completed property transactions can be stored as contracts.
Test:
Retrieve contract details with the related property, customer, and agent.


### 5. Low Offer Elicitation Scenario

The system supports detecting offers that are significantly lower than the property price.
Example:
A customer submits an offer of 3,000,000 for a property priced at 5,000,000.
Since the offer is below the defined threshold, the system can trigger the elicitation process to request additional information.


## Edge Cases

### 1. Customer Without Transactions

The database supports customers who are registered but have no appointments, offers, or contracts.
Example:
A newly registered customer can exist in the Customer table without any related records.


### 2. Property Without Offers

The database supports properties that are available but have not received any offers yet.
Example:
A newly listed property can exist without records in the Offer table.


### 3. Different Property Statuses

Properties can have different states:

- Available
- Pending
- Sold
- Withdrawn

This allows the system to track the current condition of each property.


### 4. Foreign Key Validation

The relationships between tables were tested to ensure that invalid references cannot be added.
Example:
A property cannot be created with a non-existing owner or agent.


## Database Files

The database folder contains:

- schema.sql : Creates the database tables and relationships.
- seed.sql : Inserts sample data for testing normal and edge cases.
- database.sqlite : SQLite database file.
- erd.jpg : Entity Relationship Diagram.