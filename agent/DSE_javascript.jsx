
const toAlphaChar = (idx) => String.fromCharCode(idx + 97);
const fromAlphaChar = (char) => char.charCodeAt(0) - 97;
var zeroTo25 = [...Array(26)].map((u, i) => i);
var rows = zeroTo25.map(rIdx => zeroTo25.map(cIdx => toAlphaChar((rIdx + cIdx) % 26)));

const encode = (plainText, key) => plainText.split('').map((char, textIdx) =>
{
    var keyChar = key[textIdx % key.length];
    var keyCharIdx = fromAlphaChar(keyChar);
    var charIdx = fromAlphaChar(char);
    return toAlphaChar((charIdx + keyCharIdx) % 26);
}).join('')

const decode = (ceipherText, key) => ceipherText.split('').map((char, textIdx) =>
{
    var keyChar = key[textIdx % key.length];
    var keyCharIdx = fromAlphaChar(keyChar);
    var charIdx = fromAlphaChar(char);
    return toAlphaChar((charIdx + 26 - keyCharIdx) % 26);
}).join('')

// print ENCODE('hund', 'kat') to console
$.writeln(encode('hund', 'kat'));