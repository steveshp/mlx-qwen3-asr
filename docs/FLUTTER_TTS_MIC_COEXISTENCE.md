# Flutter TTS + Microphone 동시 사용 가이드

## 핵심 원리

소프트웨어 오디오 출력(TTS)은 마이크 입력으로 들어오지 않는다.
출력 DAC와 입력 ADC는 OS 레벨에서 완전히 분리된 경로다.

Flutter에서 문제가 되는 건 **소리 섞임이 아니라 오디오 세션 충돌** — TTS 재생 시 OS가 오디오 세션을 전환하면서 마이크 녹음이 중단되는 것이다.

## 해결: 오디오 세션을 명시적으로 설정

### iOS — AVAudioSession

```dart
// audio_session 패키지 사용
import 'package:audio_session/audio_session.dart';

Future<void> configureAudioSession() async {
  final session = await AudioSession.instance;
  await session.configure(AudioSessionConfiguration(
    avAudioSessionCategory: AVAudioSessionCategory.playAndRecord,
    avAudioSessionCategoryOptions: {
      AVAudioSessionCategoryOption.defaultToSpeaker,
      AVAudioSessionCategoryOption.allowBluetooth,
      AVAudioSessionCategoryOption.allowBluetoothA2DP,
    },
    avAudioSessionMode: AVAudioSessionMode.voiceChat,
  ));
  await session.setActive(true);
}
```

**필수 사항**:
- 카테고리는 반드시 `.playAndRecord` — `.playback`이나 `.record` 단독 사용 시 동시 동작 불가
- `defaultToSpeaker`: 이어폰 없을 때 스피커로 출력 (안 하면 수화기 스피커로 나옴)
- `allowBluetooth`: 블루투스 이어폰/헤드셋 지원
- 앱 시작 시 **한 번만** 설정, TTS/녹음 시작할 때마다 바꾸지 않는다

### Android — AudioManager

```dart
// audio_session 패키지가 Android도 처리
Future<void> configureAudioSession() async {
  final session = await AudioSession.instance;
  await session.configure(AudioSessionConfiguration(
    androidAudioAttributes: const AndroidAudioAttributes(
      contentType: AndroidAudioContentType.speech,
      usage: AndroidAudioUsage.voiceCommunication,
    ),
    androidAudioFocusGainType: AndroidAudioFocusGainType.gain,
    androidWillPauseWhenDucked: false,
  ));
  await session.setActive(true);
}
```

**필수 사항**:
- `usage: voiceCommunication` — 동시 입출력 허용
- `androidWillPauseWhenDucked: false` — 다른 앱이 오디오 포커스 요청해도 녹음 유지

### 통합 설정 (iOS + Android)

```dart
Future<void> configureAudioSession() async {
  final session = await AudioSession.instance;
  await session.configure(AudioSessionConfiguration(
    // iOS
    avAudioSessionCategory: AVAudioSessionCategory.playAndRecord,
    avAudioSessionCategoryOptions: {
      AVAudioSessionCategoryOption.defaultToSpeaker,
      AVAudioSessionCategoryOption.allowBluetooth,
      AVAudioSessionCategoryOption.allowBluetoothA2DP,
    },
    avAudioSessionMode: AVAudioSessionMode.voiceChat,
    // Android
    androidAudioAttributes: const AndroidAudioAttributes(
      contentType: AndroidAudioContentType.speech,
      usage: AndroidAudioUsage.voiceCommunication,
    ),
    androidAudioFocusGainType: AndroidAudioFocusGainType.gain,
    androidWillPauseWhenDucked: false,
  ));
  await session.setActive(true);
}
```

## 필요한 패키지

```yaml
# pubspec.yaml
dependencies:
  audio_session: ^0.1.21       # 오디오 세션 설정 (iOS + Android)
  record: ^5.1.2               # 마이크 녹음
  flutter_tts: ^4.0.2          # TTS 재생
```

## 구현 순서

### 1. 앱 시작 시 오디오 세션 설정

```dart
class MyApp extends StatefulWidget { ... }

class _MyAppState extends State<MyApp> {
  @override
  void initState() {
    super.initState();
    configureAudioSession(); // 앱 시작 시 한 번
  }
}
```

### 2. TTS 초기화

```dart
final FlutterTts tts = FlutterTts();

Future<void> initTts() async {
  await tts.setLanguage('ko-KR');
  await tts.setSpeechRate(0.5);

  // iOS: 공유 오디오 세션 사용 (세션을 뺏지 않도록)
  await tts.setIosAudioCategory(
    IosTextToSpeechAudioCategory.ambient,
    [IosTextToSpeechAudioCategoryOptions.mixWithOthers],
  );
}
```

### 3. 마이크 녹음

```dart
final AudioRecorder recorder = AudioRecorder();

Future<void> startRecording() async {
  if (await recorder.hasPermission()) {
    await recorder.start(
      const RecordConfig(
        encoder: AudioEncoder.pcm16bits,
        sampleRate: 16000,
        numChannels: 1,
      ),
      path: tempFilePath,
    );
  }
}
```

### 4. 동시 실행

```dart
// 녹음 먼저 시작
await startRecording();

// TTS는 녹음 중에도 자유롭게 호출
await tts.speak('안녕하세요, 무엇을 도와드릴까요?');

// 녹음은 계속 진행 중 — TTS와 독립
```

## 흔한 실수와 해결

| 증상 | 원인 | 해결 |
|------|------|------|
| TTS 시작하면 녹음 중단 | 오디오 세션 미설정 또는 `.playback` 카테고리 | `.playAndRecord`로 변경 |
| 소리가 수화기 스피커로 나옴 | `defaultToSpeaker` 옵션 누락 | 옵션 추가 |
| 블루투스 이어폰에서 안 됨 | `allowBluetooth` 옵션 누락 | 옵션 추가 |
| TTS 후 녹음 품질 저하 | TTS가 세션 카테고리를 변경 | TTS에 `mixWithOthers` 설정 |
| Android에서 간헐적 녹음 중단 | 오디오 포커스 뺏김 | `androidWillPauseWhenDucked: false` |

## iOS Info.plist

```xml
<key>NSMicrophoneUsageDescription</key>
<string>음성 인식을 위해 마이크 접근이 필요합니다</string>
```

## Android AndroidManifest.xml

```xml
<uses-permission android:name="android.permission.RECORD_AUDIO" />
```

## 스피커 사용 시 (이어폰 없을 때)

이어폰 없이 스피커로 TTS를 재생하면, 소리가 공기를 통해 물리적으로 마이크에 도달한다.
이 경우 Software AEC(Acoustic Echo Cancellation)가 필요하다.

- iOS: `AVAudioSessionMode.voiceChat` 설정 시 시스템 AEC 자동 활성화
- Android: `usage: voiceCommunication` 설정 시 시스템 AEC 자동 활성화

위 설정을 이미 적용했으므로, **스피커 모드에서도 시스템 AEC가 동작한다**.
완벽하지는 않지만 대부분의 경우 충분하다. 이어폰 사용이 가장 확실한 해결책이다.
