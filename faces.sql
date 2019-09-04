BEGIN TRANSACTION;
DROP TABLE IF EXISTS "faces";
CREATE TABLE IF NOT EXISTS "faces" (
	"id"	INTEGER PRIMARY KEY AUTOINCREMENT,
	"face_location"	BLOB,
	"face_encoding"	BLOB,
	"imagepath"	TEXT,
	"face"	BLOB
);
COMMIT;
