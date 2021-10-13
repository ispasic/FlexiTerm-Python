CREATE TABLE IF NOT EXISTS data_document
(
  id  		VARCHAR(30),
  document  TEXT,
  verbatim  TEXT,
  PRIMARY KEY(id)
);
CREATE TABLE IF NOT EXISTS data_sentence
(
  id				VARCHAR(50),
  doc_id			VARCHAR(30),
  position			INT,
  sentence			TEXT,
  tagged_sentence	TEXT,
  tags				TEXT,
  PRIMARY KEY(id)
);
CREATE TABLE IF NOT EXISTS data_token
(
  sentence_id	VARCHAR(50),
  position		INT,
  token			VARCHAR(30),
  stem			VARCHAR(30),
  lemma			VARCHAR(30),
  tag			VARCHAR(10),
  gtag			VARCHAR(10),
  wntag			CHAR(1),
  FOREIGN KEY(sentence_id) REFERENCES data_sentence(id)
);
CREATE TABLE IF NOT EXISTS term_phrase
(
  id			VARCHAR(60),
  sentence_id	VARCHAR(50),
  token_start	INT,
  token_length	INT,
  phrase		TEXT,
  normalised	TEXT,
  flat			TEXT,
  PRIMARY KEY(id),
  FOREIGN KEY(sentence_id) REFERENCES data_sentence(id)
);
CREATE TABLE IF NOT EXISTS stopword
(
  word	VARCHAR(30),
  PRIMARY KEY(word)
);
CREATE TABLE IF NOT EXISTS token
(
  token  VARCHAR(30),
  PRIMARY KEY(token)
);
CREATE TABLE IF NOT EXISTS token_similarity
(
  token1  VARCHAR(30),
  token2  VARCHAR(30),
  PRIMARY KEY(token1, token2),
  FOREIGN KEY(token1) REFERENCES token(token),
  FOREIGN KEY(token2) REFERENCES token(token)
);
CREATE TABLE IF NOT EXISTS term_bag
(
  id	INT,
  token	VARCHAR(30),
  FOREIGN KEY(id) REFERENCES term_normalised(rowid)
);
CREATE TABLE IF NOT EXISTS term_normalised
(
  normalised	TEXT,
  expanded		TEXT,
  len			INT,
  PRIMARY KEY(normalised)
);
CREATE TABLE IF NOT EXISTS term_nested
(
  parent	TEXT,
  child		TEXT,
  PRIMARY KEY(parent, child)
);
CREATE TABLE IF NOT EXISTS term_nested_aux
(
  parent INT,
  child  INT,
  PRIMARY KEY(parent, child),
  FOREIGN KEY(child)  REFERENCES term_normalised(rowid),
  FOREIGN KEY(parent) REFERENCES term_normalised(rowid)
);
CREATE TABLE IF NOT EXISTS term_termhood
(
  expanded			TEXT,
  representative	TEXT,
  len				INT,
  f					INT,
  s					INT,
  nf				INT,
  c					REAL,
  PRIMARY KEY(expanded)
);
CREATE TABLE IF NOT EXISTS term_acronym
(
  acronym		TEXT NOT NULL,
  phrase		TEXT,
  normalised	TEXT,
  PRIMARY KEY(acronym)
);
CREATE TABLE IF NOT EXISTS term_output
(
  id      INT  NOT NULL,
  variant TEXT NOT NULL,
  c       REAL NOT NULL,
  f       INT  NOT NULL,
  df      INT,
  idf     REAL,
  
  PRIMARY KEY (id, variant)
);
CREATE TABLE IF NOT EXISTS tmp_acronym
(
  acronym		TEXT NOT NULL,
  phrase		TEXT,
  normalised	TEXT
);
CREATE TABLE IF NOT EXISTS tmp_normalised
(
  changefrom	TEXT NOT NULL,
  changeto		TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS output_label
(
  doc_id	TEXT,
  start		INT,
  offset	INT,
  label		INT,
  PRIMARY KEY(doc_id, start, offset, label)
);

CREATE TABLE IF NOT EXISTS v1
(
  lemma	VARCHAR(30),
  value	INT,
  PRIMARY KEY(lemma)
);
CREATE TABLE IF NOT EXISTS v2
(
  lemma	VARCHAR(30),
  value	INT,
  PRIMARY KEY(lemma)
);
DELETE FROM data_document;
DELETE FROM data_sentence;
DELETE FROM data_token;
DELETE FROM stopword;
DELETE FROM term_acronym;
DELETE FROM term_bag;
DELETE FROM term_normalised;
DELETE FROM term_nested_aux;
DELETE FROM term_nested;
DELETE FROM term_phrase;
DELETE FROM term_termhood;
DELETE FROM token;
DELETE FROM token_similarity;
DELETE FROM output_label;
DELETE FROM tmp_acronym;
DELETE FROM tmp_normalised;
DROP INDEX IF EXISTS idx01;
DROP INDEX IF EXISTS idx02;
DROP INDEX IF EXISTS idx03;
DROP INDEX IF EXISTS idx04;
DROP INDEX IF EXISTS idx05;
DROP INDEX IF EXISTS idx06;
DROP INDEX IF EXISTS idx07;
DROP INDEX IF EXISTS idx08;
DROP INDEX IF EXISTS idx09;
DROP INDEX IF EXISTS idx10;
DROP INDEX IF EXISTS idx11;
DROP INDEX IF EXISTS idx12;
DROP INDEX IF EXISTS idx13;
DROP INDEX IF EXISTS idx14;
DROP INDEX IF EXISTS idx15;
DROP INDEX IF EXISTS idx16;
DROP INDEX IF EXISTS idx17;