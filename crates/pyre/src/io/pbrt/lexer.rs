//! Lexer for the PBRT scene-description format. Returns a flat stream of
//! `Token`s; the parser layer interprets them into directives.
//!
//! Tokens:
//! - Identifiers — unquoted keywords like `LookAt`, `Camera`, `WorldBegin`.
//! - Strings — `"..."`. PBRT uses these both for positional string args
//!   (`Camera "perspective"`) and for parameter headers (`"float fov"`).
//! - Numbers — integers and floats. PBRT doesn't distinguish at lex
//!   time; the parser coerces based on parameter type, but we keep an
//!   `Integer` token so `"integer indices"` arrays don't lose precision.
//! - Brackets — `[` and `]` delimit parameter value arrays.
//!
//! Whitespace and `#`-to-end-of-line comments are skipped.

use std::fmt;

#[derive(Debug, Clone)]
pub enum Token {
    Identifier(String),
    String(String),
    Integer(i64),
    Float(f64),
    LBracket,
    RBracket,
}

#[derive(Debug, Clone, Copy)]
pub struct Pos {
    pub line: u32,
    pub col: u32,
}

impl fmt::Display for Pos {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "line {}, col {}", self.line, self.col)
    }
}

#[derive(Debug, Clone)]
pub struct Spanned {
    pub token: Token,
    pub pos: Pos,
}

#[derive(Debug, thiserror::Error)]
pub enum LexError {
    #[error("{pos}: unterminated string literal")]
    UnterminatedString { pos: Pos },
    #[error("{pos}: unexpected character {ch:?}")]
    UnexpectedChar { pos: Pos, ch: char },
    #[error("{pos}: invalid number {text:?}")]
    InvalidNumber { pos: Pos, text: String },
}

pub fn tokenize(input: &str) -> Result<Vec<Spanned>, LexError> {
    let mut out = Vec::new();
    let bytes = input.as_bytes();
    let mut i = 0;
    let mut line: u32 = 1;
    let mut line_start: usize = 0;

    while i < bytes.len() {
        let pos = Pos {
            line,
            col: (i - line_start) as u32 + 1,
        };
        let c = bytes[i];
        match c {
            b' ' | b'\t' | b'\r' => {
                i += 1;
            }
            b'\n' => {
                line += 1;
                i += 1;
                line_start = i;
            }
            b'#' => {
                // Comment to end of line.
                while i < bytes.len() && bytes[i] != b'\n' {
                    i += 1;
                }
            }
            b'[' => {
                out.push(Spanned {
                    token: Token::LBracket,
                    pos,
                });
                i += 1;
            }
            b']' => {
                out.push(Spanned {
                    token: Token::RBracket,
                    pos,
                });
                i += 1;
            }
            b'"' => {
                i += 1;
                let start = i;
                while i < bytes.len() && bytes[i] != b'"' {
                    if bytes[i] == b'\n' {
                        return Err(LexError::UnterminatedString { pos });
                    }
                    i += 1;
                }
                if i >= bytes.len() {
                    return Err(LexError::UnterminatedString { pos });
                }
                let s = std::str::from_utf8(&bytes[start..i])
                    .expect("PBRT strings must be UTF-8")
                    .to_string();
                out.push(Spanned {
                    token: Token::String(s),
                    pos,
                });
                i += 1; // closing quote
            }
            _ if is_number_start(c) => {
                let start = i;
                while i < bytes.len() && is_number_body(bytes[i]) {
                    i += 1;
                }
                let text = &input[start..i];
                let token = if text.contains('.') || text.contains('e') || text.contains('E') {
                    let v: f64 = text
                        .parse()
                        .map_err(|_| LexError::InvalidNumber { pos, text: text.to_string() })?;
                    Token::Float(v)
                } else {
                    match text.parse::<i64>() {
                        Ok(v) => Token::Integer(v),
                        Err(_) => {
                            // Fall back to float — e.g. `1e6`-ish forms that
                            // happened to lack a `.`.
                            let v: f64 = text.parse().map_err(|_| LexError::InvalidNumber {
                                pos,
                                text: text.to_string(),
                            })?;
                            Token::Float(v)
                        }
                    }
                };
                out.push(Spanned { token, pos });
            }
            _ if is_ident_start(c) => {
                let start = i;
                while i < bytes.len() && is_ident_body(bytes[i]) {
                    i += 1;
                }
                let s = input[start..i].to_string();
                out.push(Spanned {
                    token: Token::Identifier(s),
                    pos,
                });
            }
            _ => {
                return Err(LexError::UnexpectedChar { pos, ch: c as char });
            }
        }
    }
    Ok(out)
}

fn is_number_start(c: u8) -> bool {
    c.is_ascii_digit() || c == b'-' || c == b'+' || c == b'.'
}

fn is_number_body(c: u8) -> bool {
    c.is_ascii_digit() || c == b'-' || c == b'+' || c == b'.' || c == b'e' || c == b'E'
}

fn is_ident_start(c: u8) -> bool {
    c.is_ascii_alphabetic() || c == b'_'
}

fn is_ident_body(c: u8) -> bool {
    c.is_ascii_alphanumeric() || c == b'_'
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn tokenizes_lookat() {
        let src = "LookAt 3 4 1.5 .5 .5 0 0 0 1";
        let toks = tokenize(src).unwrap();
        assert!(matches!(toks[0].token, Token::Identifier(ref s) if s == "LookAt"));
        assert_eq!(toks.len(), 10);
    }

    #[test]
    fn tokenizes_parameter_pair() {
        let toks = tokenize(r#"Camera "perspective" "float fov" [39]"#).unwrap();
        assert!(matches!(toks[0].token, Token::Identifier(ref s) if s == "Camera"));
        assert!(matches!(toks[1].token, Token::String(ref s) if s == "perspective"));
        assert!(matches!(toks[2].token, Token::String(ref s) if s == "float fov"));
        assert!(matches!(toks[3].token, Token::LBracket));
        assert!(matches!(toks[4].token, Token::Integer(39)));
        assert!(matches!(toks[5].token, Token::RBracket));
    }

    #[test]
    fn skips_comments() {
        let toks = tokenize("# only a comment\nLookAt 1 2 3 0 0 0 0 1 0").unwrap();
        assert!(matches!(toks[0].token, Token::Identifier(ref s) if s == "LookAt"));
    }
}
