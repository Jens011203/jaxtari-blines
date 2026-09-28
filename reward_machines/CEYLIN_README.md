# Ceylin'in 5 oyunu: breakout, mspacman, gravitar, montezumarevenge, skiing

## Neyin doğrulandığı, neyin doğrulanmadığı

Her oyunun `NUM_FEATURES` sayısı ve alan indeksleri **gerçek jaxatari ortamı
çalıştırılarak** (`reward_machines/tools/dump_layout.py`) doğrulandı, elle
tahmin edilmedi. Bu yüzden indeksler güvenilir.

**Doğrulanmayan / v1 olan şey: RM tasarımının kendisi** (hangi olaylara ne
kadar ödül verileceği, hangi olayların önemli olduğu). Bunlar ilk varsayım,
1M adımlık koşulardan sonra ayarlanacak.

## Önemli kısıt: bazı oyunlarda skor/can gözlemde yok

| Oyun | Gözlemde skor/can var mı? | RM ne görebiliyor |
|---|---|---|
| breakout | ✅ (lives, score) | blok kırılması, can kaybı, duvar temizlenmesi |
| gravitar | can var, skor yok | yakıt tankı toplama, düşman yok etme, can kaybı |
| skiing | can yok (gerekmiyor), `successful_gates` sayacı var | kapı geçişi |
| mspacman | ❌ sadece Info'da | pellet/power-pellet yeme |
| montezumarevenge | ❌ sadece Info'da | eşya toplama (hangi eşya olduğu ayırt edilemiyor) |

Ms. Pacman ve Montezuma için RM, ölüm veya oda/seviye değişimini
**göremiyor** -- bunlar `Observation`'da yok, sadece `Info`'da. Ortamın kendi
`done` sinyali zaten ölümde bölümü bitiriyor, RM'nin bunu ayrıca bilmesine
gerek yok. Ama "anahtar al -> kapı aç -> yeni oda" gibi zengin bir alt-hedef
zinciri şu anki gözlemle **kurulamıyor**. Bu, raporun RQ2 (RM tasarımı
oyunlar arasında sorunsuz transfer oluyor mu?) sorusuna dürüst ve değerli bir
cevap: hayır, çünkü bazı oyunların gözlem tasarımı buna izin vermiyor.

Bunu düzeltmenin tek yolu jaxatari'nin kendi `_get_observation`
fonksiyonuna `room_id`/`lives`/`score` eklemek -- bu jaxatari'ye (üçüncü
parti bağımlılık) yapılacak bir değişiklik, sadece RM dosyalarına değil.
Yapmak isterseniz önce ekiple (Jens, ders ekibi) konuşun, çünkü bu tüm
oyunun gözlem boyutunu değiştirir ve zaten eğitilmiş her şeyi etkiler.

## Sırada ne var

1. `uv run pytest tests/test_rm.py -k "breakout or mspacman or gravitar or montezumarevenge or skiing" -v`
   çalıştır. Özellikle `test_random_rollout_report` çıktısına bak: hangi
   transition hiç ateşlenmemiş, ödül aralığı mantıklı mı.
2. Sorun varsa (bir olay hiç tetiklenmiyor, işaret yanlış), önce
   `reward_machines/tools/dump_layout.py <oyun>` ile indeksleri tekrar
   doğrula, sonra ilgili `*_rm.py` dosyasındaki index sabitlerini düzelt.
3. Testler temizse 1M adımlık koşulara geç:
   ```
   uv run python main.py alg=ppo_oc ENV_ID=breakout GAME_RM=breakout TOTAL_TIMESTEPS=1000000
   uv run python main.py alg=ppo_oc ENV_ID=breakout GAME_RM=null       TOTAL_TIMESTEPS=1000000
   ```
   (baseline için `GAME_RM=null`, diğer 4 oyun için `ENV_ID`/`GAME_RM`'i
   değiştir: `mspacman`, `gravitar`, `montezumarevenge`, `skiing`)
4. Sonuçlara göre `TRANSITIONS` listesindeki ödül büyüklüklerini ayarla.
