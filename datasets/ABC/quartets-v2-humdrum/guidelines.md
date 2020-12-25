# Annotation Guidelines

## Ziel

Das Ziel des Annotationsprojekts liegt in der Untersuchung harmonischer Tonalität zwischen ca. 1500 und der Gegenwart. Zu diesem Zweck sollen umfangreiche Repertoires unterschiedlicher stilistischer und historischer Provenienz nach einem vorab festgelegten, maschinenlesbaren Annotationsstandard analysiert werden. Die Analysen sollen in der Folge mit Hilfe computerwissenschaftlicher Methoden statistisch ausgewertet werden.

## Philosophie

Die Annotatoren sollten auf größtmögliche Konsistenz achten, sowohl was die Analyse (analoge Situationen sollten möglichst analog annotiert werden) als auch die Handhabung des Standards angeht; dennoch geht es nicht um die Imitation von automatisierter Analyse, da die Kontextabhängigkeit der Einzelakkorde Berücksichtigung finden sollte. Menschliche Intuitionen und "Auffassung" dürfen in die Analyse einfließen. Dies betrifft vor allem die Annotation von harmonisch unterspezifizierten Texturen (z.B. monophoner Passagen, Unisono-Texturen oder implizite bzw. latente Polyphonie). Allerdings ist von großflächigeren Deutungen etwa im Sinne einer Schenkerschen Prolongation abzusehen.

## Prozedur

Da die in der Folge dargelegten Annotationsrichtlinien im Verlauf des Projekts weiter verfeinert werden sollen, müssen sämtliche Probleme, die beim Annotieren auftreten, von den Annotatoren explizit im [Forum](https://github.com/DCMLab/ABC/issues) ("Issues") festgehalten werden.

1. **Symboleingabe über MuseScore (Software-Umgebung)**: Die Annotation soll direkt mit MuseScore (kostenlos als [Download](https://musescore.org/de/download) erhältlich) über dem jeweiligen System, als Akkordsymbol, eingetragen werden. Dazu wird zunächst eine Note zum Zeitpunkt des Akkordwechsels selektiert. Mit `Strg+K` springt der Cursor dann in die Akkordsymbolzeile. Navigiert werden kann von dort aus mit der Leertaste zum nächsten Notenereignis, sowie mit dem Tabulator zum nächsten Takt. In Ausnahmefällen kann das Akkordsymbol auch an andere Stimmen angebunden werden. Es muss dann aber manuell mit der Maus in die oberste Zeile gezogen werden. Aus Gründen der besseren Kontrolle ist es wünschenswert, dass die Annotatoren zugleich auch mit einer gängigen Printausgabe der Partitur arbeiten. Fehler im digitalen Notentext (im Vergleich zur gedruckten Notenausgabe) können punktuell korrigiert werden.
2. **Tonart und Tonartwechsel**: Unmittelbar vor dem ersten Akkordsymbol muss die **Tonart** des Stückes angegeben werden. Dur-Tonarten werden mit Großbuchstaben, Moll-Tonarten mit Kleinbuchstaben notiert. Es ist die internationale Notation zu verwenden (`B` statt `H`; `Bb` für das deutsche `B`). Beispiele: `Bb.I`, `f#.i` oder `F.V`. **Modulation** kann hier pragmatisch als "Notationsabkürzung" verstanden werden: Ab dem Moment, wo die Notation zu aufwendig werden würde (wenn z.B. zu viele Zwischendominanten angenommen werden müssten), sollte eine neue Tonart angezeigt werden. Notiert wird dies mit einem Stufensymbol und einem darauf folgenden Punkt. Das Stufensymbol zeigt die jeweilige Tonart in Bezug zur Grundtonart des Stückes an (z.B. nach erfolgter Modulation in die Dominanttonart: `V.`). Dies wird einmalig zum Zeitpunkt des Tonartenwechsels zusammen mit dem jeweiligen Akkordsymbol notiert. Damit ist klar ausgedrückt, dass sich das Stück nun in einer neuen Tonart befindet. Sofern möglich, sollten Annotatoren eine Modulation zum frühest möglichen Zeitpunkt ansetzen, auch unter Berücksichtigung des nachfolgenden Kontextes. Alle Tonarten sind auf die Grundtonart zu beziehen. Beispiel: `V.IV` (wir befinden uns in der Dominanttonart, es erklingt die IV. Stufe).
3. **Akkordstufen und -typen**: Möglichst jedem Akkord sollte eine **Akkordstufe** zugeordnet werden. Dur-Akkorde werden mit groß geschriebenen Stufen bezeichnet (`I`, `II`, `III`, `IV` etc.), Moll-Akkorde mit Kleinschreibung (`i`, `ii`, `iii`, `iv`etc.). Übermäßige Akkorde werden mit großen Stufen und darauf folgendem `+`, verminderte und halbverminderte mit kleinen Stufen und darauf folgendem `o` bzw. `%` notiert. Akkorde in Grundstellung mit großer Septime werden mit `M7`notiert. **Wichtig:** Um eine konsistente Datenauswertung zu ermöglichen, dürfen ausschließlich die hier aufgeführten Akkordsymbole, sowie die Spezialsymbole `.Ger6`, `.Fr6` und `.It6`, verwendet werden (siehe unten). Akkorden, deren Grundtöne nicht der jeweils vorliegenden Diatonik entstammen (Dur bzw. natürliches Moll), wird ein `#` bzw. `b`vorangestellt, z.B.: `#IV` oder `bVII`. Sind darüber hinaus weitere Akkordsymbole notwendig, muss dies vor Verwendung im [Forum](https://github.com/DCMLab/ABC/issues) abgeklärt werden.
4. **Spezialsymbole** sollen nur im Falle der "augmentierten (übermäßigen) Sextakkorde" - (Quint)-Sextakkord und Terzquartakkord - verwendet werden (`.Ger6`, `.It6`, `.Fr6`). Der Neapolitaner hingegen wird als `.bII` kodiert. Wichtig: Steht das Spezialsymbol am Anfang eines Ausdrucks, muss dieser mit einem `.` eingeleitet werden; steht er innerhalb eines Ausdruck, ist dies nicht notwendig.
5. **Umkehrungen** werden im Anschluss an das Stufensymbol ohne Klammern notiert. So können Sie von Vorhalten und hinzugefügten Noten (siehe unten) unterschieden werden. Für Dreiklänge sind `6` (erste Umkehrung) und `64` (zweite Umkehrung) möglich. Für Vierklänge können `7` (Grundstellung), `65` (erste Umkehrung), `43` (zweite Umkehrung) und `2` (dritte Umkehrung) verwendet werden.
6. **Vorhalte** werden in runden Klammern `(`, `)` notiert. **Wichtig:** Vorhalte beziehen sich ausdrücklich auf den Grundton (bzw. die Stufe), nicht auf den Basston! Beispielsweise könnte eine Akkordfolge mit aufgelöstem Vorhalt wie folgt aussehen: `V(64) V I`.
7. **Hinzugefügte Töne** werden mit einem `+`versehen und ebenfalls in runden Klammern notiert, z.B. `I(+4+2)` oder `I(+42)`. Das Plus bezieht sich dabei immer nur auf die unmittelbar folgende Ziffer. Im ersten Beispiel sind beide Töne hinzugefügt, im zweiten Beispiel ist die `2`ein Vorhalt.
8. **Zwischendominanten** werden mit einem Slash (`/`) notiert. Beispielsweise wird ein A-Dur-Dreiklang in C-Dur, der sich als Dominante auf einen D-Moll-Dreiklang bezieht, als `V/ii` notiert.
9. **Orgelpunkte** werden mit `"` notiert, z.B. `V"I ii V I"` (eine harmonische Bewegung über der V. Stufe). Ein harmonisches Pendel über einem Tonika-Orgelpunkt sollte bevorzugt als `I"I I(64) I"` bezeichnet werden (wenn möglich nicht als `I IV64 I`). **Wichtig:** Es ist auf Konsistenz in der Notation zu achten! Bei einem **doppelten Orgelpunkt** (i.d. Regel eine Bordunquinte) wird das Stufensymbol gemäß dem tieferen Ton gewählt und dazu noch eine Ziffer für das diatonische Intervall hinzugefügt (also z.B. bei G-D in C-Dur: `V5`; bei D-F im C-Kontext wäre es dementsprechend `II5`). Dabei hat das Stufensymbol keine tongeschlechtliche Implikation.
10. **Ambiguität**: Es ist grundsätzlich möglich, Mehrdeutigkeiten (plausible multiple Analysen) festzuhalten. Ambiguitäten in der Akkordinterpretation können mit dem Trennsymbol `-` ausgedrückt werden, z.B. `V7-viio` oder `V(64)-I64` (ohne Leerzeichen). Vorzuziehen ist allerdings stets die wahrscheinlichere bzw. plausiblere Deutung.
11. **Phrasengrenzen**: Neben Akkorden sollen auch Endpunkte von Kadenzvorgängen durch ein Doppel-Backslash `\\`annotiert werden. Beispielsequenz: `ii6 V I\\ IV6`. Elisionen werden ebenfalls mit Doppelbackslash markiert, welcher von der jeweiligen Stufe umschlossen wird: z.B. `I\\I`.
12. Sollte eine harmonische Interpretation für einen bestimmten Abschnitt unangemessen sein, wird folgendes Symbol benutzt: `@none`.