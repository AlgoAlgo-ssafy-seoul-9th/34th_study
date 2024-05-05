import java.io.IOException;
import java.util.*;
import java.io.*;
public class 민웅 {
    static int N, K;
    static Integer[] exercise;
    static boolean[] visited;
    static int cnt = 0;
    public static void main(String[] args) throws IOException {

        BufferedReader br = new BufferedReader(new InputStreamReader(System.in));
        StringTokenizer st = new StringTokenizer(br.readLine());

        N = Integer.parseInt(st.nextToken());
        K = Integer.parseInt(st.nextToken());

        visited = new boolean[N];
        exercise = new Integer[N];

        st = new StringTokenizer(br.readLine());
        for (int i = 0; i<N; i++) {
            exercise[i] = Integer.parseInt(st.nextToken());
        }
//        System.out.println(Arrays.toString(exercise));
        Arrays.sort(exercise, Collections.reverseOrder());
//        System.out.println(Arrays.toString(exercise));

        permutation(0,500);
        System.out.println(cnt);
    }

    private static void permutation(int level, int score) {
        if (level == N){
            cnt++;
            return;
        }

        for (int i=0; i<N; i++){
            if(!visited[i]) {
                int tmp = score + exercise[i] - K;
                if (tmp < 500) {
                    break;
                }
                visited[i] = true;
                permutation(level+1, tmp);
                visited[i] = false;
            }
        }
    }
}